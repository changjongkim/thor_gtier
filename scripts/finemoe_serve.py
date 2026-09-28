#!/usr/bin/env python3
"""FineMoE (Yu et al., EuroSys'26; github.com/IntelliSys-Lab/FineMoE-EuroSys26)
on the same prompts, timed per request like the other runners.  Its released
code with the Qwen3-MoE port (third_party/finemoe_qwen3_sm110.patch).

The GPU expert cache gets `cache_size` = budget / expert size slots (FineMoE's
own knob; memcal matches its measured peak to PHASOR's).  Expert maps (its
semantic embeddings and expert trajectories) are built by --collect from
workloads other than the evaluated one and loaded with --maps, so no evaluated
request is seen in advance (as for DuoServe*'s predictor).

usage: finemoe_serve.py --checkpoint DIR --workload W.json --budget-gib B --maps DIR --out O.json
       finemoe_serve.py --checkpoint DIR --collect OUTDIR --workload W1.json [W2.json ...] [--limit N]
"""
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--workload", required=True, nargs="+")
ap.add_argument("--budget-gib", type=float, default=0.0)
ap.add_argument("--maps")
ap.add_argument("--collect")
ap.add_argument("--prefetch-distance", type=int, default=6)
ap.add_argument("--store-capacity", type=int, default=1000)
ap.add_argument("--device-memory-ratio", type=float, default=0.0, help="nominal: FineMoE's own sizing (0 = use --budget-gib)")
ap.add_argument("--max-prompt", type=int, default=8192)
ap.add_argument("--max-new", type=int, default=32)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out")
a = ap.parse_args()
sys.path.insert(0, "/home/thor/kcj/thor_gtier/scripts")
from memwatch import MemWatch
_mw = MemWatch()
import torch
from transformers import AutoConfig, AutoTokenizer
from finemoe import MoE

cfg = AutoConfig.from_pretrained(a.checkpoint)
expert_bytes = 3 * cfg.hidden_size * cfg.moe_intermediate_size * 2
work = [w for p in a.workload for w in json.load(open(p))]
if a.limit: work = work[:a.limit]
opts = dict(prefetch_distance=a.prefetch_distance, store_capacity=a.store_capacity)
if a.collect:
    opts.update(collect_maps=True, eval_mode="online", trace_capacity=max(1, len(work)))
elif a.device_memory_ratio > 0:
    opts.update(device_memory_ratio=a.device_memory_ratio)
else:
    opts.update(cache_size=max(1, int(a.budget_gib * (1 << 30) // expert_bytes)))
tok = AutoTokenizer.from_pretrained(a.checkpoint)
t0 = time.time()
model = MoE(a.checkpoint, opts)
if a.maps: model.engine.expert_map_store.import_store_data(a.maps)
model.warmup()
load_s = time.time() - t0


class Clock:
    def __init__(self): self.t = []
    def put(self, v): torch.cuda.synchronize(); self.t.append(time.time())
    def end(self): pass


rows = []
for w in work:
    ids = tok(w["prompt"], return_tensors="pt").input_ids[:, :a.max_prompt]
    new = min(a.max_new, int(w.get("max_new", a.max_new)))
    st0 = model.engine.cache.stats
    clk = Clock(); torch.cuda.synchronize(); ts = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=new, min_new_tokens=new, do_sample=False,
                             attention_mask=torch.ones_like(ids), pad_token_id=tok.eos_token_id, streamer=clk)
    torch.cuda.synchronize(); te = time.time()
    st1 = model.engine.cache.stats
    gen = clk.t[1:]
    h, m = st1.hits - st0.hits, st1.misses - st0.misses
    rows.append({"name": w["name"], "prompt_tok": int(ids.shape[1]), "new_tok": int(out.shape[1] - ids.shape[1]),
                 "ttft_s": (gen[0] - ts) if gen else te - ts,
                 "tpot_ms": ((gen[-1] - gen[0]) / (len(gen) - 1) * 1e3) if len(gen) > 1 else 0.0,
                 "request_s": te - ts, "expert_hit_rate": h / (h + m) if h + m else None,
                 "out_ids": out[0, ids.shape[1]:].tolist()})
    print("REQ " + json.dumps({k: v for k, v in rows[-1].items() if k != "out_ids"}), flush=True)
if a.collect:
    if model.engine.predictor.dropped_updates:
        print("map collection dropped updates", file=sys.stderr); os._exit(2)
    os.makedirs(a.collect, exist_ok=True)
    model.engine.expert_map_store.export_store_data(a.collect)
    print(f"COLLECTED {len(rows)} prompts -> {a.collect}", flush=True)
n = len(rows)
res = {"system": "finemoe", "budget_gib": a.budget_gib, "cache_size": opts.get("cache_size"),
       "load_s": load_s, "requests": n,
       "ttft_s": sum(r["ttft_s"] for r in rows) / n, "tpot_ms": sum(r["tpot_ms"] for r in rows) / n,
       "request_s": sum(r["request_s"] for r in rows) / n, "peak_gib": _mw.peak_gib(), "rows": rows}
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"RESULT policy=finemoe budget={a.budget_gib:.2f} requests={n} ttft_s={res['ttft_s']:.4f} "
          f"tpot_ms={res['tpot_ms']:.3f} request_s={res['request_s']:.4f} peak_gib={res['peak_gib']:.2f} compute=measured")
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
