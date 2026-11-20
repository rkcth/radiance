#!/usr/bin/env python3
"""chain_scalar_audit.py BUILD_DIR OUT_DIR [--check]

A chained wait ends with an acquire: the agent-scope fence for the vector caches and, unless the kernel's wait is the
variant that leaves it out, `s_dcache_inv` for the scalar data cache (libr4d/r4d_chain.h). A kernel may use that
variant only when nothing it reads behind the wait goes through a scalar load. This audit compiles every HIP unit that
holds a chained kernel to gfx1201 assembly with the build's own flags (compile_commands.json) and checks each chained
(`r4dchain`) kernel: one that has no `s_dcache_inv` must have no scalar load, from a base other than the kernel
argument segment (s[0:1]), behind its in-kernel acquire (`global_inv` at device or system scope).

--list also lists the kernels that keep it, with their scalar-load counts.
--check exits 1 when a kernel breaks that rule. Without it the kernels that skip the instruction are listed."""
import json, sys, os, re, shlex, subprocess, concurrent.futures as cf

args = [a for a in sys.argv[1:] if not a.startswith("--")]
bd, out = os.path.abspath(args[0]), os.path.abspath(args[1])
check = "--check" in sys.argv
os.makedirs(out, exist_ok=True)
cc = json.load(open(bd + "/compile_commands.json"))
units = [e for e in cc if e["file"].endswith(".hip") and
         ("/libr4d/" in e["file"] or "/core/device/" in e["file"] or "/core/sched/" in e["file"])]


def sregs(t):
    """The scalar registers an operand names, [] for anything else (v, exec, vcc, an immediate)."""
    m = re.fullmatch(r"s\[(\d+):(\d+)\]", t)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    m = re.fullmatch(r"s(\d+)", t)
    return [int(m.group(1))] if m else []


def run(e):
    a = shlex.split(e["command"])
    o = []
    i = 0
    while i < len(a):
        if a[i] == "-o":
            i += 2; continue
        if a[i] == "-c":
            i += 1; continue
        if a[i].startswith("--offload-arch=") and a[i] != "--offload-arch=gfx1201":
            i += 1; continue
        if a[i] == "--no-offload-compress":
            i += 1; continue
        o.append(a[i]); i += 1
    s = out + "/" + os.path.basename(e["file"]) + ".s"
    o += shlex.split(os.environ.get("AUDIT_FLAGS", ""))
    if "--reuse" in sys.argv and os.path.exists(s):
        return s
    o = [x for x in o if x != e["file"]] + ["--offload-device-only", "-S", "-o", s, e["file"]]
    r = subprocess.run(o, cwd=e["directory"], capture_output=True, text=True)
    if r.returncode:
        print("FAIL", e["file"], r.stderr[-400:])
        return None
    return s


with cf.ThreadPoolExecutor(int(os.environ.get("J", "24"))) as ex:
    sfs = [s for s in ex.map(run, units) if s]
bad = nk = skip = keep = 0
for fn in sorted(sfs):
    cur = None
    rows = []
    for line in open(fn, errors="replace"):
        m = re.match(r"^(_Z\S+):\s*; @", line)
        if m:
            cur = {"name": m.group(1), "inv": 0, "dc": 0, "after": 0, "ex": [], "k": {0, 1}}
            rows.append(cur)
            continue
        if cur is None:
            continue
        s = line.strip()
        if not s or s.startswith((".", ";")):
            continue
        if s.startswith("global_inv") and ("SCOPE_DEV" in s or "SCOPE_SYS" in s):
            cur["inv"] += 1
        if s.startswith("s_dcache_inv"):
            cur["dc"] += 1
        ops = [o.strip() for o in s.split(None, 1)[1].split(",")] if " " in s else []
        op = s.split()[0]
        if op.startswith(("s_load_", "s_buffer_load_")) and len(ops) >= 2:
            if op.startswith("s_buffer"):
                base = None
            else:
                base = sregs(ops[1])
            if cur["inv"] > 0 and not (base and all(r in cur["k"] for r in base)):
                cur["after"] += 1
                if len(cur["ex"]) < 2:
                    cur["ex"].append(s)
            for r in sregs(ops[0]):
                cur["k"].discard(r)
        elif ops and sregs(ops[0]) and not op.startswith(("s_cmp", "s_bitcmp", "s_wait", "s_cbranch")):
            d = sregs(ops[0])
            srcs = [sregs(o) for o in ops[1:]]
            src0 = srcs[0] if srcs else []
            derived = False
            if op in ("s_mov_b64", "s_mov_b32") and src0:
                derived = all(r in cur["k"] for r in src0)
            elif op in ("s_add_nc_u64", "s_add_u32", "s_add_co_u32", "s_add_nc_u32", "s_add_co_ci_u32", "s_addc_u32") and src0:
                derived = all(r in cur["k"] for r in src0)
            elif op == "s_cselect_b32" and src0:
                derived = all(r in cur["k"] for r in src0)
            if derived:
                cur["k"].update(d)
            elif op not in ("s_cselect_b32",) or True:
                for r in d:
                    cur["k"].discard(r)
    for r in rows:
        if "r4dchain" not in r["name"]:
            continue
        nk += 1
        if r["dc"]:
            keep += 1
            if "--list" in sys.argv:
                print("keeps s_dcache_inv: %-90s scalar loads behind the acquire: %d" % (r["name"][:90], r["after"]))
            continue
        skip += 1
        if r["after"]:
            bad += 1
        print("%s skips s_dcache_inv: %-90s scalar loads behind the acquire: %d %s" %
              ("BAD" if r["after"] else "ok ", r["name"][:90], r["after"], r["ex"]))
print("chained kernels %d: keep s_dcache_inv %d, skip it %d, of those with a scalar load behind the acquire %d" %
      (nk, keep, skip, bad))
sys.exit(1 if (check and bad) else 0)
