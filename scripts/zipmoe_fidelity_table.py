#!/usr/bin/env python3
"""results/ZIPMOE_FIDELITY -> FIDELITY.md (paper 4.1)."""
import glob, json, os, re
F = "/home/thor/kcj/thor_gtier/results/ZIPMOE_FIDELITY"


def official(fp, alg):
    fs = sorted(glob.glob(f"{F}/official/ZipMoE-qwen-M{fp}-B1-L64-C{alg}-*.json"))
    if not fs: return None
    d = json.load(open(fs[-1]))["data"]
    e2e = [x for m in d for x in m["E2E"]]; ttft = [x for m in d for x in m["TTFT"]]
    tpot = [t for m in d for s in m["TPOT"] for t in s]
    return {"n": len(e2e), "e2e": sum(e2e) / len(e2e), "ttft": sum(ttft) / len(ttft),
            "tpot_ms": 1e3 * sum(tpot) / max(len(tpot), 1)}


out = ["# ZipMoE fidelity on its own model and harness (paper 4.1)", "",
       "Model: Qwen1.5-MoE-A2.7B-Chat (the ZipMoE paper's model). Harness: ZipMoE's `evaluation/evaluate.py`,",
       "24 ShareGPT prompts (its sampler, seed 321), 512-token prompts, 64 new tokens, 30 s cooling per prompt,",
       "after its warm-up prompt. Footprint presets are fractions of a 64 GB Orin; scaled by 64/122.8 so a",
       "footprint means the same bytes here (`ZIPMOE_MEM_SCALE`). No page-cache scrubbing (its native setting).", "",
       "## (a) Its caching against its LRU / LFU options (mean end-to-end s per request)", "",
       "| footprint | ZipMoE | LRU | LFU | ZipMoE vs LRU | ZipMoE vs LFU |", "|---:|---:|---:|---:|---:|---:|"]
for fp in (10, 20, 30):
    r = {a: official(fp, a) for a in ("ZipMoE", "LRU", "LFU")}
    if not r["ZipMoE"]: continue
    c = lambda a: f"{r[a]['e2e']:.2f} ({r[a]['n']})" if r[a] else "-"
    rel = lambda a: f"{(r['ZipMoE']['e2e'] / r[a]['e2e'] - 1) * 100:+.1f}%" if r[a] else "-"
    out.append(f"| {fp} GB | {c('ZipMoE')} | {c('LRU')} | {c('LFU')} | {rel('LRU')} | {rel('LFU')} |")
out += ["", "## (b) Our runner against its harness, same prompts and memory (20 GB)", ""]
o = official(20, "ZipMoE")
p = f"{F}/ours_M20.json"
if o and os.path.exists(p):
    d = json.load(open(p)); rows = d["rows"]
    ours = {"e2e": d["request_s"], "ttft": d["ttft_s"], "tpot_ms": d["tpot_ms"]}
    ours_w = {"e2e": sum(r["request_s"] for r in rows[1:]) / (len(rows) - 1)}   # without the cold first request
    out += ["| | end-to-end s | TTFT s | TPOT ms |", "|---|---:|---:|---:|",
            f"| ZipMoE harness | {o['e2e']:.2f} | {o['ttft']:.2f} | {o['tpot_ms']:.0f} |",
            f"| our runner | {ours['e2e']:.2f} | {ours['ttft']:.2f} | {ours['tpot_ms']:.0f} |",
            f"| our runner, first request excluded (its harness warms up) | {ours_w['e2e']:.2f} | | |", "",
            f"Difference (warm): {(ours_w['e2e'] / o['e2e'] - 1) * 100:+.1f}%."]
print("\n".join(out))
