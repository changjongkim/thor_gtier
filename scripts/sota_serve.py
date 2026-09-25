#!/usr/bin/env python3
"""Serve a workload's prompts through a real offloading system and time each
request the way the serving driver reports it: time to first token, time per
output token, whole-request time.

The prompts are the ones the routing traces were captured from (same
truncation, same number of new tokens), so a row here and a row from
serve_bench describe the same requests.

Systems:
  moe-infinity   EAM-guided prefetch and caching, SSD -> host -> GPU
"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--system", required=True, choices=["moe-infinity"])
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--workload", required=True)
ap.add_argument("--offload-dir", required=True)
ap.add_argument("--budget-gib", type=float, required=True,
                help="device memory the system may use for experts")
ap.add_argument("--max-prompt", type=int, default=8192)
ap.add_argument("--max-new", type=int, default=32)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

import torch
from transformers import AutoTokenizer
sys.path.insert(0, "/home/thor/kcj/thor_gtier/scripts")
from memwatch import MemWatch
_mw = MemWatch()

def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0

def read_bytes():
    for line in open("/proc/self/io"):
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    return 0

class Clock:
    """A streamer that only notes when each token arrives."""
    def __init__(self): self.t = []
    def put(self, v): self.t.append(time.time())
    def end(self): pass

tok = AutoTokenizer.from_pretrained(a.checkpoint)
work = json.load(open(a.workload))
if a.limit: work = work[:a.limit]

m0 = mem_avail_gib()
t0 = time.time()
if a.system == "moe-infinity":
    from moe_infinity import MoE
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    ratio = min(0.95, a.budget_gib / total)
    model = MoE(a.checkpoint, {"offload_path": a.offload_dir,
                               "device_memory_ratio": ratio})
load_s = time.time() - t0

rows = []
for w in work:
    ids = tok(w["prompt"], return_tensors="pt").input_ids[:, :a.max_prompt].to("cuda:0")
    clk = Clock()
    torch.cuda.synchronize()
    rb0 = read_bytes(); ts = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=a.max_new, min_new_tokens=a.max_new,
                             do_sample=False, pad_token_id=tok.eos_token_id,
                             streamer=clk)
    torch.cuda.synchronize()
    te = time.time()
    # put() is called once with the prompt, then once per new token
    gen = clk.t[1:]
    n_new = out.shape[1] - ids.shape[1]
    ttft = (gen[0] - ts) if gen else te - ts
    tpot = ((gen[-1] - gen[0]) / (len(gen) - 1)) if len(gen) > 1 else 0.0
    rows.append({"name": w["name"], "prompt_tok": int(ids.shape[1]), "new_tok": int(n_new),
                 "ttft_s": ttft, "tpot_ms": tpot * 1e3, "request_s": te - ts,
                 "read_gib": (read_bytes() - rb0) / 2**30})
    print(json.dumps(rows[-1]), flush=True)

n = len(rows)
res = {"system": a.system, "checkpoint": a.checkpoint, "workload": a.workload,
       "budget_gib": a.budget_gib, "load_s": load_s,
       "footprint_gib": m0 - mem_avail_gib(),
       "requests": n,
       "ttft_s": sum(r["ttft_s"] for r in rows) / n,
       "tpot_ms": sum(r["tpot_ms"] for r in rows) / n,
       "request_s": sum(r["request_s"] for r in rows) / n,
       "rows": rows}
res["peak_gib"] = _mw.peak_gib()
json.dump(res, open(a.out, "w"), indent=1)
print(f"RESULT policy={a.system} budget={a.budget_gib:.2f} requests={n} "
      f"ttft_s={res['ttft_s']:.4f} tpot_ms={res['tpot_ms']:.3f} "
      f"request_s={res['request_s']:.4f} footprint_gib={res['footprint_gib']:.2f} peak_gib={_mw.peak_gib():.2f}")
