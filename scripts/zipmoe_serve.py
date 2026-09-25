#!/usr/bin/env python3
"""Serve a workload's prompts through ZipMoE (Yang et al., ICML'26; code at
github.com/npnothard/ZipMoE-ICML26, ported to Qwen3-MoE) and time each request
the way the other systems are timed.  Configuration follows ZipMoE's own
evaluation/evaluate.py (LZ4HC, 3 file chunks, 6 compute threads, GPU pool 0.95);
only the memory ratio is set from the budget on this device.

usage: zipmoe_serve.py --workload W.json --budget-gib B --trace T.pt --out O.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/thor/kcj/ZipMoE")
os.chdir("/home/thor/kcj/ZipMoE")
ap = argparse.ArgumentParser()
ap.add_argument("--model-type", default="qwen3")
ap.add_argument("--workload", required=True)
ap.add_argument("--budget-gib", type=float, required=True)
ap.add_argument("--trace", required=True)
ap.add_argument("--max-prompt", type=int, default=8192)
ap.add_argument("--max-new", type=int, default=32)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

import torch
from transformers import AutoTokenizer
from entry.llm_modeling import MoE
from utils.constants import (List_expert_topk, List_num_elements_per_expert,
    List_num_tensors_per_expert, List_num_expert_layers, List_num_experts,
    List_first_k_dense_replace)

def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"): return int(line.split()[1]) / 2**20
    return 0.0

class Clock:
    def __init__(self): self.t = []
    def put(self, v): self.t.append(time.time())
    def end(self): pass

mt = a.model_type
ckpt = f"/home/thor/kcj/ZipMoE/models/{mt}/"
total = torch.cuda.get_device_properties(0).total_memory / 2**30
cfg = {
    "offload_path": f"/home/thor/kcj/ZipMoE/offload/{mt}/",
    "caching_algorithm": "ZipMoE", "prefetcher_topk": 4,
    "device_memory_ratio": min(0.95, a.budget_gib / total),
    "gpu_pool_ratio": 0.95, "batch_size": 1, "code_type": "LZ4HC",
    "hyperparam_state_margin": 0.1, "num_file_chunks": 3, "num_compute_threads": 6,
    "trace_path": a.trace,
    "expert_topk": List_expert_topk[mt],
    "num_elements_per_expert": List_num_elements_per_expert[mt],
    "num_tensors_per_expert": List_num_tensors_per_expert[mt],
    "num_expert_layers": List_num_expert_layers[mt],
    "num_experts": List_num_experts[mt],
    "first_k_dense_replace": List_first_k_dense_replace[mt],
}
m0 = mem_avail_gib(); t0 = time.time()
model = MoE(ckpt, cfg)
tok = AutoTokenizer.from_pretrained(ckpt)
load_s = time.time() - t0
work = json.load(open(a.workload))
if a.limit: work = work[:a.limit]
rows = []
for w in work:
    ids = tok(w["prompt"], return_tensors="pt").input_ids[:, :a.max_prompt].to("cuda:0")
    new = min(a.max_new, int(w.get("max_new", a.max_new)))
    clk = Clock(); torch.cuda.synchronize(); ts = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=new, min_new_tokens=new, do_sample=False,
                             attention_mask=torch.ones_like(ids),
                             pad_token_id=tok.eos_token_id, streamer=clk)
    torch.cuda.synchronize(); te = time.time()
    gen = clk.t[1:]
    ttft = (gen[0] - ts) if gen else te - ts
    tpot = ((gen[-1] - gen[0]) / (len(gen) - 1)) if len(gen) > 1 else 0.0
    rows.append({"name": w["name"], "prompt_tok": int(ids.shape[1]),
                 "new_tok": int(out.shape[1] - ids.shape[1]),
                 "ttft_s": ttft, "tpot_ms": tpot * 1e3, "request_s": te - ts})
    print("REQ " + json.dumps(rows[-1]), flush=True)
n = len(rows)
res = {"system": "ZipMoE", "budget_gib": a.budget_gib, "load_s": load_s,
       "footprint_gib": m0 - mem_avail_gib(), "requests": n,
       "ttft_s": sum(r["ttft_s"] for r in rows) / n,
       "tpot_ms": sum(r["tpot_ms"] for r in rows) / n,
       "request_s": sum(r["request_s"] for r in rows) / n, "rows": rows}
json.dump(res, open(a.out, "w"), indent=1)
print(f"RESULT policy=zipmoe budget={a.budget_gib:.2f} requests={n} ttft_s={res['ttft_s']:.4f} "
      f"tpot_ms={res['tpot_ms']:.3f} request_s={res['request_s']:.4f} "
      f"footprint_gib={res['footprint_gib']:.2f} compute=measured")
