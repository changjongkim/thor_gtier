#!/usr/bin/env python3
"""Tables for stage 3: each system on its own data path under one memory cap.
Per model and per workload, the same prompts under every system and budget."""
import glob, os
R = "/home/thor/kcj/thor_gtier/results/MATRIX3"
MODELS = ["qwen30b", "mixtral8x7b", "qwen235b"]
WL = ["longbench", "sharegpt", "mmlu"]
SYS = [("lru", "LRU (pread+copy)"), ("moeinf", "MoE-Infinity* (pread+copy, prefetch)"),
       ("mixtral", "Mixtral-offloading* (copy, speculative)"),
       ("flashmoe", "FlashMoE* (copy, learned per-layer eviction)"),
       ("duoserve", "DuoServe-MoE* (copy, predicted prefetch)"),
       ("llama", "llama.cpp (--cpu-moe, mmap)"), ("ledger", "PHASOR")]
ABL = [("ledger_0.45", "PHASOR"), ("abl_nooverlap", "- layer pipelining"),
       ("abl_noasync", "- continuous submission"), ("abl_pread", "pread+copy path instead of gTier"),
       ("abl_lru", "LRU residency on gTier path"), ("abl_mix0", "- prompt routing term"),
       ("abl_mix1", "- decode history term"), ("abl_norec", "- recency term"),
       ("abl_count", "count utility")]

def res(path):
    d = None; peak = None
    try:
        for line in open(path):
            if line.startswith("RESULT "):
                d = {k: v for k, v in (x.split("=", 1) for x in line.split()[1:])}
            if line.startswith("cgroup_peak_gib="):
                peak = line.strip().split("=")[1]
    except OSError:
        return None
    if d is not None and peak is not None: d["peak"] = peak
    return d

def g(d, k, fmt):
    return format(float(d[k]), fmt) if d and k in d else "-"

print("# Serving matrix — each system on its own data path\n")
print("Same prompts on every model. Every run under a cgroup cap of budget + 0.5 GiB "
      "(pinned host memory and page cache counted). I/O measured; per-token compute "
      "calibrated with llama-bench (results/MATRIX/calib.tsv) except llama.cpp, which is "
      "measured end to end. Budgets are fractions of the model's bytes.\n")
for m in MODELS:
    if not glob.glob(f"{R}/{m}/*/*.txt"): continue
    print(f"## {m}\n")
    for w in WL:
        if not glob.glob(f"{R}/{m}/{w}/*.txt"): continue
        print(f"### {m} / {w}\n")
        print("| budget | system | request s | TTFT s | TPOT ms | tok/s | peak GiB |")
        print("|---|---|---|---|---|---|---|")
        for fr in ["0.25", "0.45", "0.65"]:
            rows = {k: res(f"{R}/{m}/{w}/{k}_{fr}.txt") for k, _ in SYS}
            for k, name in SYS:
                d = rows[k]
                if not d: continue
                print(f"| {fr} | {name} | {g(d,'request_s','.3f')} | {g(d,'ttft_s','.3f')} | "
                      f"{g(d,'tpot_ms','.1f')} | {g(d,'throughput_tok_s','.2f')} | {d.get('peak','-')} |")
            led = rows["ledger"]
            base = [(k, d) for k, d in rows.items() if d and k != "ledger"]
            if led and base:
                bk, bd = min(base, key=lambda x: float(x[1]["request_s"]))
                print(f"| {fr} | **PHASOR vs best other ({bk})** | "
                      f"**{float(bd['request_s'])/float(led['request_s']):.2f}x** | "
                      f"{float(bd['ttft_s'])/float(led['ttft_s']):.2f}x | "
                      f"{float(bd['tpot_ms'])/float(led['tpot_ms']):.2f}x | | |")
        b = [(n, res(f"{R}/{m}/{w}/{k}.txt")) for k, n in
             (("moeinf_b4_0.45", "MoE-Infinity* batch 4"), ("flashmoe_b4_0.45", "FlashMoE* batch 4"), ("duoserve_b4_0.45", "DuoServe-MoE* batch 4"),
              ("ledger_b4_0.45", "PHASOR batch 4"))]
        if any(d for _, d in b):
            print(f"\nBatch 4 at 0.45:\n\n| system | request s | TTFT s | TPOT ms | tok/s |\n|---|---|---|---|---|")
            for n, d in b:
                if d: print(f"| {n} | {g(d,'request_s','.3f')} | {g(d,'ttft_s','.3f')} | {g(d,'tpot_ms','.1f')} | {g(d,'throughput_tok_s','.2f')} |")
        a = [(n, res(f"{R}/{m}/{w}/{k}.txt")) for k, n in ABL]
        if sum(1 for _, d in a if d) > 1:
            print(f"\nAblation at 0.45:\n\n| configuration | request s | TTFT s | TPOT ms |\n|---|---|---|---|")
            for n, d in a:
                if d: print(f"| {n} | {g(d,'request_s','.3f')} | {g(d,'ttft_s','.3f')} | {g(d,'tpot_ms','.1f')} |")
        print()
