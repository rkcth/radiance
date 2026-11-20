#!/usr/bin/env python3
"""vhead.py -- a vocabulary-weighted lm_head checkpoint, as the IQR-V4 recipes convert it.

    vhead.py [--keep-lm-head] SRC DST CLASSES

SRC is a Hugging Face checkpoint directory (Qwen/Qwen3.8-27B), CLASSES a width-class file from
data/vhead/ (q38-27b-vhead-4.0.cls, -4.5, -5.0: the head's mean code width). DST becomes SRC with
lm_head.weight replaced by its width classes; every other file of SRC is linked, not copied. Pass DST
to rad-convert in place of SRC. DST must be absent or empty and on the same filesystem as the shards.
With --keep-lm-head, also index the original lm_head.weight so a native MTP drafter can
quantise its own copy without duplicating the source shard.

A vocabulary-weighted head stores each vocabulary row on one of four trellis widths, the tokens that
carry the most probability on the widest. The classes were chosen per 256-row run to minimise each
row's output error weighted by its token's mean probability on calibration text, at the file's mean
width; every TP2 rank's half of the vocabulary holds the same number of rows of each class, so the
class tensors shard by rows with the vocabulary. DST/vhead.safetensors holds, for each width B:

  lm_head.kB.weight  [rows, n_embd] bf16  the class's rows of lm_head.weight
  lm_head.kB.ids     [rows, 1] i32        each row's vocabulary id

rows in ascending id within each half of the vocabulary, and lm_head.vrow [n_vocab, 1] i32: an id's
width in the top four bits and its row in the class tensor below. rad-convert names the classes
output.kB; a recipe quantises each on its own width, and GPTQ reads the head's input Hessian for all
of them (libquant pools output.kB to output).

CLASSES: one ASCII line `radiance-vhead-1 n_vocab=V widths=a,b,c,d`, then V/4 bytes, an id's class
(the index of its width in `widths`) two bits each, id 4i in the low bits of byte i.

bf16 is moved as raw u16 (no arithmetic happens here), so numpy alone suffices.
"""
import argparse
import errno
import json
import os
import struct
import tempfile

import numpy as np


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def st_tensor(path, name):
    """One tensor of a safetensors file as raw bytes viewed as u16 (bf16) or i32, shaped."""
    hdr, base = st_header(path)
    m = hdr[name]
    dt = {"BF16": np.uint16, "F16": np.uint16, "I32": np.int32}[m["dtype"]]
    a, b = m["data_offsets"]
    return np.fromfile(path, dtype=dt, count=(b - a) // np.dtype(dt).itemsize,
                       offset=base + a).reshape(m["shape"]), m["dtype"]


def st_write(path, tensors):
    """tensors: [(name, array, dtype string)], written in that order, 8-byte aligned header."""
    hdr, off = {}, 0
    for name, a, dt in tensors:
        hdr[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        off += a.nbytes
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for _, a, _ in tensors:
            f.write(np.ascontiguousarray(a).tobytes())


def read_classes(path):
    with open(path, "rb") as f:
        head = f.readline().decode().split()
        if not head or head[0] != "radiance-vhead-1":
            raise SystemExit(f"vhead.py: {path} is not a radiance-vhead-1 class file")
        kv = dict(t.split("=", 1) for t in head[1:])
        V, widths = int(kv["n_vocab"]), [int(w) for w in kv["widths"].split(",")]
        packed = np.frombuffer(f.read(), dtype=np.uint8)
    if packed.size != (V + 3) // 4:
        raise SystemExit(f"vhead.py: {path} holds {packed.size} bytes, {(V + 3) // 4} expected")
    cls = np.stack([(packed >> (2 * j)) & 3 for j in range(4)], 1).reshape(-1)[:V]
    return cls.astype(np.int64), widths


def main():
    parser = argparse.ArgumentParser(description="Prepare a vocabulary-weighted checkpoint.")
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("classes")
    parser.add_argument("--keep-lm-head", action="store_true",
                        help="keep the original lm_head.weight indexed for a separate draft head")
    args = parser.parse_args()
    src, dst, cpath = os.path.abspath(args.src), os.path.abspath(args.dst), args.classes
    src_real, dst_real = os.path.realpath(src), os.path.realpath(dst)
    if os.path.commonpath([src_real, dst_real]) == src_real:
        parser.error("DST must be separate from SRC (not inside it or an alias for it)")
    if os.path.lexists(dst) and (os.path.islink(dst) or not os.path.isdir(dst) or os.listdir(dst)):
        parser.error("DST must be absent or an empty directory")
    cls, widths = read_classes(cpath)
    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    W, dt = st_tensor(os.path.join(src, idx["weight_map"]["lm_head.weight"]), "lm_head.weight")
    V = W.shape[0]
    if cls.size != V or V % 2:
        raise SystemExit(f"vhead.py: the class file covers {cls.size} ids, the head has {V} rows")
    half = V // 2
    ids = np.arange(V, dtype=np.int64)
    vrow = np.full((V, 1), -1, dtype=np.int32)
    out_ids, out_w = [], []
    for i, k in enumerate(widths):
        rows = np.concatenate([ids[:half][cls[:half] == i], ids[half:][cls[half:] == i]])
        vrow[rows, 0] = (k << 28) | np.arange(rows.size, dtype=np.int32)
        out_ids.append((f"lm_head.k{k}.ids", rows.astype(np.int32)[:, None], "I32"))
        out_w.append((f"lm_head.k{k}.weight", W[rows], dt))
    assert (vrow >= 0).all()
    wm = dict(idx["weight_map"])
    if not args.keep_lm_head:
        del wm["lm_head.weight"]
    for name, _, _ in out_ids + out_w + [("lm_head.vrow", None, None)]:
        wm[name] = "vhead.safetensors"
    # Build beside DST and publish only once the head, index, and all links exist. A failed copy,
    # link or rename must not leave a checkpoint whose index points at missing or stale shards.
    parent = os.path.dirname(dst)
    os.makedirs(parent, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".vhead-", dir=parent) as tmp:
        stage = os.path.join(tmp, "checkpoint")
        os.mkdir(stage)
        keep = set(idx["weight_map"].values())
        for f in os.listdir(src):
            if f == "model.safetensors.index.json" or (f.endswith(".safetensors") and f not in keep):
                continue
            if f == "vhead.safetensors":
                raise SystemExit("vhead.py: SRC already indexes vhead.safetensors; use the original checkpoint")
            source, target = os.path.join(src, f), os.path.join(stage, f)
            if f in keep:
                # The converter rejects external shard symlinks. Link, don't copy the model.
                try:
                    os.link(os.path.realpath(source), target)
                except OSError as exc:
                    if exc.errno == errno.EXDEV:
                        raise SystemExit("vhead.py: SRC shards and DST must share a filesystem "
                                         "for hard links") from exc
                    if exc.errno in (errno.EPERM, errno.EACCES):
                        raise SystemExit(f"vhead.py: cannot hard-link {source}: the shard must be "
                                         "yours or writable by you (fs.protected_hardlinks)") from exc
                    raise
            else:
                os.symlink(source, target)
        st_write(os.path.join(stage, "vhead.safetensors"),
                 out_ids + [("lm_head.vrow", vrow, "I32")] + out_w)
        with open(os.path.join(stage, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": idx.get("metadata", {}), "weight_map": wm}, f)
        # POSIX rename can replace an empty directory but not one that gained files meanwhile.
        os.rename(stage, dst)
    print(f"vhead.py: {dst}: " + " ".join(f"k{k}:{int((cls == i).sum())}" for i, k in enumerate(widths))
          + f", mean {sum(widths[i] * int((cls == i).sum()) for i in range(len(widths))) / V:.3f} b")


if __name__ == "__main__":
    main()
