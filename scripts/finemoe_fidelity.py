#!/usr/bin/env python3
"""FineMoE's own timing harness (demo/eval.py measure(): request context, manual
greedy loop over its engine) on N prompts at a given cache, for comparison with
our runner.  usage: finemoe_fidelity.py CKPT WORKLOAD MAPS BUDGET_GIB N OUT"""
import json, os, sys
sys.path.insert(0, "/home/thor/kcj/FineMoE-EuroSys26")
import torch
from transformers import AutoConfig, AutoTokenizer
from finemoe import MoE
from demo.eval import measure
ck, wl, maps, budget, n, out = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
cfg = AutoConfig.from_pretrained(ck)
eb = 3 * cfg.hidden_size * cfg.moe_intermediate_size * 2
tok = AutoTokenizer.from_pretrained(ck)
model = MoE(ck, dict(cache_size=max(1, int(budget * (1 << 30) // eb))))
model.engine.expert_map_store.import_store_data(maps)
model.warmup()
rows = []
for w in json.load(open(wl))[:n]:
    ids = tok(w["prompt"], return_tensors="pt").input_ids[:, :8192].to(model.engine.device)
    new = int(w.get("max_new", 32))
    r = measure(model, ids, torch.ones_like(ids), new)
    r["request_s"] = (r["ttft_ms"] + r["tpot_ms"] * (new - 1)) / 1000; rows.append(r)
    print(json.dumps(r), flush=True)
json.dump({"harness": "FineMoE demo/eval.py measure()", "request_s": sum(r["request_s"] for r in rows) / len(rows),
           "ttft_s": sum(r["ttft_ms"] for r in rows) / len(rows) / 1000,
           "tpot_ms": sum(r["tpot_ms"] for r in rows) / len(rows), "rows": rows}, open(out, "w"), indent=1)
sys.stdout.flush(); os._exit(0)
