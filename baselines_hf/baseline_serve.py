#!/usr/bin/env python3
"""Serve a workload through a reimplemented baseline (FlashMoE*, DuoServe*) on
the transformers stack, timed exactly as phasor_serve.py times PHASOR-HF.

usage: baseline_serve.py --system flashmoe|duoserve --checkpoint DIR --workload W.json
       --budget-gib B --out O.json [--weights F.txt] [--predictor P.pt --trace T.npz ...]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--system", required=True, choices=["flashmoe", "duoserve"])
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--workload", required=True)
ap.add_argument("--budget-gib", type=float, required=True)
ap.add_argument("--weights")
ap.add_argument("--predictor")
ap.add_argument("--trace", nargs="*", default=[])
ap.add_argument("--max-prompt", type=int, default=8192)
ap.add_argument("--max-new", type=int, default=32)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=1, help="E3: serve requests in groups of this size")
ap.add_argument("--out", required=True)
a = ap.parse_args()
import torch
import offload_hf
sys.path.insert(0, "/home/thor/kcj/thor_gtier/scripts")
from memwatch import MemWatch
_mw = MemWatch()

def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"): return int(line.split()[1]) / 2**20
    return 0.0

class Clock:
    def __init__(self): self.t = []
    def put(self, v): self.t.append(time.time())
    def end(self): pass

m0 = mem_avail_gib(); t0 = time.time()
model, tok, cache = offload_hf.build(a.checkpoint, a.system, a.budget_gib, weights=a.weights,
                                     predictor=a.predictor, traces=a.trace)
load_s = time.time() - t0
work = json.load(open(a.workload))
if a.limit: work = work[:a.limit]
rows = []
if a.batch > 1:
    import batchgen
    rows = batchgen.serve(model.generate, tok, work, a.batch, a.max_prompt, a.max_new,
                          lambda: {"read_gib": cache.r.bytes_read / 2**30})
    work = []
for w in work:
    ids = tok(w["prompt"], return_tensors="pt").input_ids[:, :a.max_prompt].to("cuda:0")
    new = min(a.max_new, int(w.get("max_new", a.max_new)))
    b0 = cache.r.bytes_read
    clk = Clock(); torch.cuda.synchronize(); ts = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=new, min_new_tokens=new, do_sample=False,
                             attention_mask=torch.ones_like(ids), pad_token_id=tok.eos_token_id, streamer=clk)
    torch.cuda.synchronize(); te = time.time()
    gen = clk.t[1:]
    ttft = (gen[0] - ts) if gen else te - ts
    tpot = ((gen[-1] - gen[0]) / (len(gen) - 1)) if len(gen) > 1 else 0.0
    rows.append({"name": w["name"], "prompt_tok": int(ids.shape[1]), "new_tok": int(out.shape[1] - ids.shape[1]),
                 "ttft_s": ttft, "tpot_ms": tpot * 1e3, "request_s": te - ts,
                 "read_gib": (cache.r.bytes_read - b0) / 2**30})
    print("REQ " + json.dumps(rows[-1]), flush=True)
n = len(rows)
res = {"system": a.system, "budget_gib": a.budget_gib, "load_s": load_s, "footprint_gib": m0 - mem_avail_gib(),
       "requests": n, "ttft_s": sum(r["ttft_s"] for r in rows) / n, "tpot_ms": sum(r["tpot_ms"] for r in rows) / n,
       "request_s": sum(r["request_s"] for r in rows) / n, "rows": rows}
res["peak_gib"] = _mw.peak_gib()
if hasattr(cache, "dec_hits"):
    res["cache"] = {"hits": cache.hits, "misses": cache.misses, "decode_hits": cache.dec_hits, "decode_misses": cache.dec_misses}
res["batch"] = a.batch
if a.batch > 1: res.update(batchgen.summary(rows))
json.dump(res, open(a.out, "w"), indent=1)
print(f"RESULT policy={a.system} budget={a.budget_gib:.2f} requests={n} ttft_s={res['ttft_s']:.4f} "
      f"tpot_ms={res['tpot_ms']:.3f} request_s={res['request_s']:.4f} footprint_gib={res['footprint_gib']:.2f} peak_gib={_mw.peak_gib():.2f} compute=measured")
