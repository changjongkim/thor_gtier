#!/usr/bin/env python3
"""Tables for the architecture-level matrix (results/MATRIX5)."""
import json, os
R = "/home/thor/kcj/thor_gtier/results/MATRIX5"
MODELS = [("qwen30b", "Qwen3-30B-A3B bf16", 57.0), ("mixtral8x7b", "Mixtral-8x7B bf16", 87.0)]
WL = [("longbench", "LongBench"), ("sharegpt", "ShareGPT"), ("mmlu", "MMLU")]
SYS = [("phasor", "PHASOR"), ("zipmoe", "ZipMoE"), ("moeinf", "MoE-Infinity"), ("flashmoe", "FlashMoE*"),
       ("duoserve", "DuoServe*"), ("fiddler", "Fiddler"), ("mixoff", "Mixtral-offloading (2-bit)")]
ABL = [("lru", "LRU residency"), ("count", "count utility"), ("copy", "host-staged copy path"),
       ("nopipe", "no intra-layer pipeline"), ("noprompt", "no prompt-routing term")]

def load(prefix):
    j = prefix + ".json"; t = prefix + ".txt"
    if os.path.exists(t) and "NORUN" in open(t).read(): return "norun"
    if not os.path.exists(j): return None
    try: return json.load(open(j))
    except Exception: return None

def cell(d):
    if d == "norun": return "cannot run (OOM under cap)"
    if not d: return "-"
    return f"{d['request_s']:.2f} / {d['ttft_s']:.2f} / {d['tpot_ms']:.0f} / {d.get('peak_gib', float('nan')):.1f}"

print("# Architecture-level matrix\n\nEach cell: request s / TTFT s / TPOT ms / peak memory GiB (MemAvailable drop).")
print("Every run under a cgroup cap of budget + 0.5 GiB; `*` = reimplemented (no released code).\n")
for m, mname, gb in MODELS:
    if not os.path.isdir(f"{R}/{m}"): continue
    print(f"## {mname}\n")
    for w, wname in WL:
        print(f"### {wname}\n")
        fr = ["0.25", "0.45", "0.65", "1.08"]
        print("| system | " + " | ".join(f"{float(f):.0%} ({gb*float(f):.1f} GiB)" for f in fr) + " |")
        print("|---" * (len(fr) + 1) + "|")
        for k, name in SYS:
            row = [cell(load(f"{R}/{m}/{w}/{k}_{f}")) for f in fr]
            if all(c == "-" for c in row): continue
            print(f"| {name} | " + " | ".join(row) + " |")
        rows = [(n, load(f"{R}/{m}/{w}/abl_{k}")) for k, n in ABL]
        base = load(f"{R}/{m}/{w}/phasor_0.45")
        if base and base != "norun" and any(d for _, d in rows):
            print(f"\nAblation at 45% (request s, relative to PHASOR {base['request_s']:.2f} s):\n")
            for n, d in rows:
                if d and d != "norun":
                    print(f"- {n}: {d['request_s']:.2f} s ({d['request_s']/base['request_s']:.2f}x)")
        print()
    print("### Smallest budget each system runs at (MMLU)\n")
    for k, name in SYS:
        ok = [f for f in ["1.08", "0.65", "0.45", "0.25", "0.20", "0.15", "0.10", "0.05"]
              if (d := load(f"{R}/{m}/mmlu/{k}_{f}")) and d != "norun"]
        if ok: print(f"- {name}: {float(min(ok, key=float)):.0%}")
    print()
