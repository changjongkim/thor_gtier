#!/usr/bin/env python3
"""Where each system's request time goes: prompt compute, prefill I/O left
exposed after overlap, decode compute, decode I/O left exposed; and the bytes
each system reads.  From the RESULT lines of stage 3; no new runs."""
import sys
R = "/home/thor/kcj/thor_gtier/results"
cal = {l.split()[0]: (float(l.split()[1]), float(l.split()[2]))
       for l in open(f"{R}/MATRIX/calib.tsv") if not l.startswith("model")}
SYS = [("lru", "LRU"), ("moeinf", "MoE-Infinity*"), ("mixtral", "Mixtral-offloading*"),
       ("flashmoe", "FlashMoE*"), ("duoserve", "DuoServe*"), ("ledger", "PHASOR")]
def res(p):
    try:
        for l in open(p):
            if l.startswith("RESULT "): return dict(x.split("=", 1) for x in l.split()[1:])
    except OSError: return None
m = sys.argv[1] if len(sys.argv) > 1 else "qwen30b"
fr = sys.argv[2] if len(sys.argv) > 2 else "0.45"
cd, cp = cal[m]
print(f"## {m}, budget {fr}: request time by component (s) and bytes read\n")
print("| workload | system | request | prompt compute | exposed prefill I/O | decode compute | exposed decode I/O | prefill GiB/req | decode GiB/tok |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for w in ("longbench", "sharegpt", "mmlu"):
    for k, name in SYS:
        d = res(f"{R}/MATRIX3/{m}/{w}/{k}_{fr}.txt")
        if not d: continue
        pt, nd = float(d["prompt_tok"]), float(d["decode_tok"])
        pc = pt * cp / 1e3; ttft = float(d["ttft_s"]); tpot = float(d["tpot_ms"]) / 1e3
        dc = nd * cd / 1e3; req = float(d["request_s"])
        exp_pre = max(0.0, ttft - pc); exp_dec = max(0.0, req - ttft - dc)
        print(f"| {w} | {name} | {req:.2f} | {pc:.2f} | {exp_pre:.2f} | {dc:.2f} | {exp_dec:.2f} | "
              f"{float(d['prefill_gib']):.2f} | {float(d['decode_gib_tok']):.3f} |")
