#!/usr/bin/env python3
"""Tables for the serving matrix: per model, per workload, the same prompts
under every policy and budget.  Reads the RESULT line each run prints."""
import glob, os, re, sys
R = "/home/thor/kcj/thor_gtier/results/MATRIX"
MODELS = ["qwen30b", "mixtral8x7b", "qwen235b", "qwen30b_bf16"]
WL = ["longbench", "sharegpt", "mmlu"]
POL = [("lru", "LRU"), ("moeinf", "MoE-Infinity*"), ("mixtral", "Mixtral-offloading*"),
       ("moeinf_real", "MoE-Infinity (real system)"), ("ledger", "PHASOR")]
ABL = [("ledger_0.45", "PHASOR (full)"), ("abl_noasync", "- async submission"),
       ("abl_mix0", "- prompt routing term (history only)"),
       ("abl_mix1", "- decode history term (prompt only)"),
       ("abl_norec", "- recency term"), ("abl_nosel", "- selective admission"),
       ("abl_count", "count utility instead of estimate"),
       ("abl_profile", "+ held-out initial counts"),
       ("abl_noprefix", "- prefix pin"), ("abl_nolive", "- live-set guard"),
       ("abl_pread", "pread+copy data path"), ("ref_none", "no residency")]

def res(path):
    try:
        for line in open(path):
            if line.startswith("RESULT "):
                return {k: v for k, v in (x.split("=", 1) for x in line.split()[1:])}
    except OSError:
        pass
    return None

def f(d, k, fmt):
    return format(float(d[k]), fmt) if d and k in d else "-"

print("# Serving matrix\n")
print("Same prompts on every model; request time = prefill I/O + prompt compute + "
      "decode (I/O + compute) per token, averaged over requests.  I/O measured, compute "
      "calibrated (results/MATRIX/calib.tsv).  Budgets are fractions of the model's bytes.\n")
if os.path.exists(f"{R}/calib.tsv"):
    print("## Compute calibration\n\n```\n" + open(f"{R}/calib.tsv").read() + "```\n")
for m in MODELS:
    if not glob.glob(f"{R}/{m}/*/*.txt"):
        continue
    print(f"## {m}\n")
    for w in WL:
        if not glob.glob(f"{R}/{m}/{w}/*.txt"):
            continue
        print(f"### {m} / {w}\n")
        print("| budget | policy | request s | TTFT s | TPOT ms | tok/s (e2e) | prefill GiB/req | decode GiB/tok |")
        print("|---|---|---|---|---|---|---|---|")
        for fr in ["0.25", "0.45", "0.65"]:
            best = None; led = None
            for key, name in POL:
                d = res(f"{R}/{m}/{w}/{key}_{fr}.txt")
                if not d:
                    continue
                if key == "ledger": led = d
                elif best is None or float(d["request_s"]) < float(best["request_s"]): best = d
                print(f"| {fr} ({float(d.get('budget', 0)):.1f} GiB) | {name} | {f(d,'request_s','.3f')} | "
                      f"{f(d,'ttft_s','.3f')} | {f(d,'tpot_ms','.2f')} | {f(d,'e2e_tok_s','.3f')} | "
                      f"{f(d,'prefill_gib','.2f')} | {f(d,'decode_gib_tok','.4f')} |")
            if led and best:
                print(f"| {fr} | **PHASOR vs best baseline ({best['policy']})** | "
                      f"**{float(best['request_s'])/float(led['request_s']):.2f}x** | "
                      f"{float(best['ttft_s'])/float(led['ttft_s']):.2f}x | "
                      f"{float(best['tpot_ms'])/float(led['tpot_ms']):.2f}x | | | |")
        print()
        rows = [(n, res(f"{R}/{m}/{w}/{k}.txt")) for k, n in ABL]
        if sum(1 for _, d in rows if d) > 1:
            print(f"Ablation at 0.45 ({m} / {w}):\n")
            print("| configuration | request s | TTFT s | TPOT ms |")
            print("|---|---|---|---|")
            for n, d in rows:
                if d:
                    print(f"| {n} | {f(d,'request_s','.3f')} | {f(d,'ttft_s','.3f')} | {f(d,'tpot_ms','.2f')} |")
            print()
