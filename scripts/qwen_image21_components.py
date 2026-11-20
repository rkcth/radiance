#!/usr/bin/env python3
"""Assemble the pinned Qwen-Image 2.1 BF16 components for rad-convert.

Metadata must come from the matching Diffusers pipeline. The reference
directory holds the JSON safetensors headers for its three components.
All tensor payloads are copied.
"""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import struct
import tempfile


def read_number(stream, fmt):
    return struct.unpack("<" + fmt, stream.read(struct.calcsize(fmt)))[0]


def read_string(stream):
    size = read_number(stream, "Q")
    if size > 4096:
        raise ValueError("unexpected GGUF string length")
    return stream.read(size).decode()


def gguf_tensors(path):
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF" or read_number(stream, "I") != 3:
            raise ValueError("expected GGUF v3")
        count, fields = read_number(stream, "Q"), read_number(stream, "Q")
        if not 0 < count <= 10000 or fields > 32:
            raise ValueError("unexpected GGUF tensor or metadata count")
        metadata = {}
        for _ in range(fields):
            key = read_string(stream)
            kind = read_number(stream, "I")
            if key in metadata or kind not in (4, 8):
                raise ValueError(f"unsupported GGUF metadata {key}")
            metadata[key] = read_string(stream) if kind == 8 else read_number(stream, "I")
        if metadata.get("general.architecture") != "qwen_image21":
            raise ValueError("expected qwen_image21 GGUF")
        alignment = metadata.get("general.alignment", 32)
        if not isinstance(alignment, int) or not 1 <= alignment <= 4096 or alignment & (alignment - 1):
            raise ValueError("invalid GGUF alignment")
        tensors = {}
        for _ in range(count):
            key, rank = read_string(stream), read_number(stream, "I")
            if key in tensors or not 1 <= rank <= 4:
                raise ValueError(f"invalid GGUF tensor {key}")
            shape = [read_number(stream, "Q") for _ in range(rank)][::-1]
            kind, offset = read_number(stream, "I"), read_number(stream, "Q")
            if kind != 30 or not all(shape) or offset % alignment:
                raise ValueError(f"expected aligned BF16 tensor {key}")
            tensors[key] = (shape, offset, math.prod(shape) * 2)
        start = (stream.tell() + alignment - 1) & -alignment
    return start, tensors


def safetensors(path):
    with path.open("rb") as stream:
        size = read_number(stream, "Q")
        if not 0 < size <= 16 * 1024 * 1024:
            raise ValueError("invalid safetensors header length")
        header = json.loads(stream.read(size))
    header.pop("__metadata__", None)
    tensors = {}
    for key, item in header.items():
        shape = item["shape"]
        lo, hi = item["data_offsets"]
        if item["dtype"] != "BF16" or lo < 0 or hi - lo != math.prod(shape) * 2:
            raise ValueError(f"invalid BF16 tensor {key}")
        tensors[key] = (shape, lo, hi - lo)
    return 8 + size, tensors


def text_key(key):
    return re.sub(r"^model\.(layers\.|embed_tokens\.|norm\.)",
                  r"model.language_model.\1", key)


def vae_key(key):
    key = re.sub(r"^conv1\.", "quant_conv.", key)
    key = re.sub(r"^conv2\.", "post_quant_conv.", key)
    key = re.sub(r"^(encoder|decoder)\.conv1\.", r"\1.conv_in.", key)
    key = re.sub(r"^(encoder|decoder)\.head\.0\.", r"\1.norm_out.", key)
    key = re.sub(r"^(encoder|decoder)\.head\.2\.", r"\1.conv_out.", key)
    key = re.sub(r"\.middle\.0\.", ".mid_block.resnets.0.", key)
    key = re.sub(r"\.middle\.1\.", ".mid_block.attentions.0.", key)
    key = re.sub(r"\.middle\.2\.", ".mid_block.resnets.1.", key)
    key = re.sub(r"^encoder\.downsamples\.(\d+)\.downsamples\.2\.",
                 r"encoder.down_blocks.\1.downsampler.", key)
    key = re.sub(r"^decoder\.upsamples\.(\d+)\.upsamples\.3\.",
                 r"decoder.up_blocks.\1.upsampler.", key)
    key = re.sub(r"^encoder\.downsamples\.(\d+)\.downsamples\.(\d+)\.",
                 r"encoder.down_blocks.\1.resnets.\2.", key)
    key = re.sub(r"^decoder\.upsamples\.(\d+)\.upsamples\.(\d+)\.",
                 r"decoder.up_blocks.\1.resnets.\2.", key)
    for source, target in (("0", "norm1"), ("2", "conv1"), ("3", "norm2"), ("6", "conv2")):
        key = key.replace(f".residual.{source}.", f".{target}.")
    return key.replace(".shortcut.", ".conv_shortcut.")


def plan(path, parsed, rename, expected, reference):
    start, tensors = parsed
    mapped = {}
    end = 0
    available = path.stat().st_size - start
    for key, (shape, offset, size) in sorted(tensors.items(), key=lambda item: item[1][1]):
        name = rename(key)
        if name in mapped or offset < end or offset + size > available:
            raise ValueError(f"invalid or overlapping payload for {key}")
        end = offset + size
        if name in reference:
            target = reference[name]["shape"]
            if rename is vae_key and len(shape) == 5 and shape[2] == 1 and len(target) == 4:
                shape = shape[:2] + shape[3:]
            if shape != target:
                raise ValueError(f"{name}: shape {shape}, expected {target}")
        mapped[name] = (shape, start + offset, size)
    if mapped.keys() != expected:
        raise ValueError(f"{path.name}: missing {sorted(expected - mapped.keys())}, "
                         f"extra {sorted(mapped.keys() - expected)}")
    return mapped


def write_weights(src, dst, tensors):
    header = {}
    offset = 0
    for name, (shape, _, size) in tensors.items():
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with src.open("rb") as source, dst.open("xb") as dest:
        dest.write(struct.pack("<Q", len(encoded)))
        dest.write(encoded)
        for _, position, size in tensors.values():
            source.seek(position)
            while size:
                data = source.read(min(size, 16 * 1024 * 1024))
                if not data:
                    raise ValueError(f"truncated input {src}")
                dest.write(data)
                size -= len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("components", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("reference_headers", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true", help="validate mappings without writing")
    args = parser.parse_args()
    metadata = args.metadata
    index = json.loads((metadata / "model_index.json").read_text())
    if index.get("_class_name") != "QwenImage21Pipeline":
        parser.error("metadata is not a QwenImage21Pipeline")
    transformer = args.components / "qwen-image-2.1-UC-BF16.gguf"
    encoder = args.components / "text_encoders/qwen3vl_8b_bf16.safetensors"
    vae = args.components / "vae/qwen_image_2.1_vae_bf16.safetensors"
    jobs = []
    for component, src, read, rename, stem in (
        ("transformer", transformer, gguf_tensors, str, "diffusion_pytorch_model"),
        ("text_encoder", encoder, safetensors, text_key, "model"),
        ("vae", vae, safetensors, vae_key, "diffusion_pytorch_model"),
    ):
        shapes = json.loads((args.reference_headers / f"{component}.json").read_text())
        shapes.pop("__metadata__", None)
        tensors = plan(src, read(src), rename, set(shapes), shapes)
        print(f"{component}: {len(tensors)} BF16 tensors", flush=True)
        jobs.append((src, Path(component) / (stem + ".safetensors"), tensors))
    if args.check:
        return
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        stage = Path(temporary) / "pipeline"
        shutil.copytree(metadata, stage, ignore=shutil.ignore_patterns("*.safetensors*"))
        for src, dst, tensors in jobs:
            write_weights(src, stage / dst, tensors)
        stage.rename(args.output)


if __name__ == "__main__":
    main()
