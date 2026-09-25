#!/usr/bin/env python3
"""Mixtral-offloading (Eliseev & Mazur 2023; github.com/dvmazur/mixtral-offloading)
on the same prompts, timed per request.  Built exactly as its demo notebook:
HQQ 4-bit attention and 2-bit experts, offload_per_layer = 4, LRU expert cache
with speculative loading.  Its whole quantized model (about 17 GiB) sits in
memory; on a unified pool GPU and host parts are one pool.  Lossy: the other
systems serve bf16.

usage: mixoff_serve.py --state DIR --workload W.json --out O.json [--offload-per-layer 4]
"""
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True)
ap.add_argument("--config", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
ap.add_argument("--tokenizer", default="/home/thor/kcj/models/mixtral8x7b_bf16")
ap.add_argument("--workload", required=True)
ap.add_argument("--budget-gib", type=float, default=0)
ap.add_argument("--offload-per-layer", type=int, default=4)
ap.add_argument("--max-prompt", type=int, default=8192)
ap.add_argument("--max-new", type=int, default=32)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, "/home/thor/kcj/mixtral-offloading")
sys.path.insert(0, "/home/thor/kcj/thor_gtier/scripts")
from memwatch import MemWatch
_mw = MemWatch()
import torch
from hqq.core.quantize import BaseQuantizeConfig
from transformers import AutoConfig, AutoTokenizer
from src.build_model import OffloadConfig, QuantConfig, build_model

config = AutoConfig.from_pretrained(a.state)
E, L = config.num_local_experts, config.num_hidden_layers
opl = a.offload_per_layer
offload_config = OffloadConfig(main_size=L * (E - opl), offload_size=L * opl, buffer_size=4,
                               offload_per_layer=opl)
attn_config = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True, quant_scale=True)
attn_config["scale_quant_params"]["group_size"] = 256
ffn_config = BaseQuantizeConfig(nbits=2, group_size=16, quant_zero=True, quant_scale=True)
t0 = time.time()
model = build_model(device=torch.device("cuda:0"),
                    quant_config=QuantConfig(ffn_config=ffn_config, attn_config=attn_config),
                    offload_config=offload_config, state_path=a.state)
tok = AutoTokenizer.from_pretrained(a.tokenizer, use_fast=False)   # 4.36 tokenizers cannot parse the newer tokenizer.json
load_s = time.time() - t0

class Clock:
    def __init__(self): self.t = []
    def put(self, v): self.t.append(time.time())
    def end(self): pass

work = json.load(open(a.workload))
if a.limit: work = work[:a.limit]
rows = []
for w in work:
    ids = tok(w["prompt"], return_tensors="pt").input_ids[:, :a.max_prompt].to("cuda:0")
    new = min(a.max_new, int(w.get("max_new", a.max_new)))
    clk = Clock(); torch.cuda.synchronize(); ts = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=new, min_new_tokens=new, do_sample=False,
                             attention_mask=torch.ones_like(ids), pad_token_id=tok.eos_token_id, streamer=clk)
    torch.cuda.synchronize(); te = time.time()
    gen = clk.t[1:]
    ttft = (gen[0] - ts) if gen else te - ts
    tpot = ((gen[-1] - gen[0]) / (len(gen) - 1)) if len(gen) > 1 else 0.0
    rows.append({"name": w["name"], "prompt_tok": int(ids.shape[1]), "new_tok": int(out.shape[1] - ids.shape[1]),
                 "ttft_s": ttft, "tpot_ms": tpot * 1e3, "request_s": te - ts})
    print("REQ " + json.dumps(rows[-1]), flush=True)
n = len(rows)
res = {"system": "mixtral-offloading", "budget_gib": a.budget_gib, "load_s": load_s, "requests": n,
       "ttft_s": sum(r["ttft_s"] for r in rows) / n, "tpot_ms": sum(r["tpot_ms"] for r in rows) / n,
       "request_s": sum(r["request_s"] for r in rows) / n, "peak_gib": _mw.peak_gib(), "rows": rows}
json.dump(res, open(a.out, "w"), indent=1)
print(f"RESULT policy=mixtral-offloading budget={a.budget_gib:.2f} requests={n} ttft_s={res['ttft_s']:.4f} "
      f"tpot_ms={res['tpot_ms']:.3f} request_s={res['request_s']:.4f} peak_gib={res['peak_gib']:.2f} compute=measured")
