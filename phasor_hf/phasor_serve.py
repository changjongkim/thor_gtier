#!/usr/bin/env python3
"""Serve a workload's prompts through PHASOR-HF and time each request the way
zipmoe_serve.py and sota_serve.py time theirs (same prompts, truncation,
generation length, greedy, TTFT from the first streamed token).

usage: phasor_serve.py --checkpoint DIR --workload W.json --budget-gib B --out O.json
       [--policy phasor|lru] [--window-gib 0.5] [--slot-mib 4]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--workload", required=True)
ap.add_argument("--budget-gib", type=float, required=True)
ap.add_argument("--policy", default="phasor", help="phasor | lru | count, optionally +copy")
ap.add_argument("--no-pipeline", action="store_true")
ap.add_argument("--mix", type=float, default=0.5)
ap.add_argument("--window-gib", type=float, default=0.5)
ap.add_argument("--slot-mib", type=int, default=4)
ap.add_argument("--max-prompt", type=int, default=8192)
ap.add_argument("--max-new", type=int, default=32)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=1, help="E3: serve requests in groups of this size")
ap.add_argument("--profile", action="store_true", help="E5: per-phase wait / expert / MoE time (synchronizing)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
import torch
import phasor_hf
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
model, tok, eng = phasor_hf.build(a.checkpoint, a.budget_gib, window_gib=a.window_gib,
                                  slot_mib=a.slot_mib, policy=a.policy, mix=a.mix,
                                  pipeline=not a.no_pipeline)
load_s = time.time() - t0
work = json.load(open(a.workload))
if a.limit: work = work[:a.limit]
if a.profile:
    phasor_hf.PROFILE = {}
rows = []


if a.batch > 1:
    import batchgen
    def _st():
        x = eng.stats(); return {"hits": x[0], "misses": x[1], "read_gib": x[2] / 2**30}
    def _gen(**k):
        eng.begin_request(); return model.generate(**k)
    rows = batchgen.serve(_gen, tok, work, a.batch, a.max_prompt, a.max_new, _st)
    work = []
for w in work:
    ids = tok(w["prompt"], return_tensors="pt").input_ids[:, :a.max_prompt].to("cuda:0")
    new = min(a.max_new, int(w.get("max_new", a.max_new)))
    eng.begin_request()
    s0 = eng.stats()
    clk = Clock(); torch.cuda.synchronize(); ts = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=new, min_new_tokens=new, do_sample=False,
                             attention_mask=torch.ones_like(ids),
                             pad_token_id=tok.eos_token_id, streamer=clk)
    torch.cuda.synchronize(); te = time.time()
    s1 = eng.stats()
    gen = clk.t[1:]
    ttft = (gen[0] - ts) if gen else te - ts
    tpot = ((gen[-1] - gen[0]) / (len(gen) - 1)) if len(gen) > 1 else 0.0
    rows.append({"name": w["name"], "prompt_tok": int(ids.shape[1]),
                 "new_tok": int(out.shape[1] - ids.shape[1]), "ttft_s": ttft,
                 "tpot_ms": tpot * 1e3, "request_s": te - ts,
                 "hits": s1[0] - s0[0], "misses": s1[1] - s0[1],
                 "read_gib": (s1[2] - s0[2]) / 2**30})
    print("REQ " + json.dumps(rows[-1]), flush=True)
n = len(rows)
res = {"system": "PHASOR-HF", "policy": a.policy, "budget_gib": a.budget_gib,
       "load_s": load_s, "footprint_gib": m0 - mem_avail_gib(), "requests": n,
       "ttft_s": sum(r["ttft_s"] for r in rows) / n,
       "tpot_ms": sum(r["tpot_ms"] for r in rows) / n,
       "request_s": sum(r["request_s"] for r in rows) / n, "rows": rows}
res["peak_gib"] = _mw.peak_gib()
res["batch"] = a.batch
if a.batch > 1:
    res.update(batchgen.summary(rows))
if a.profile:
    res["profile_s"] = phasor_hf.PROFILE
json.dump(res, open(a.out, "w"), indent=1)
print(f"RESULT policy=phasor-hf-{a.policy} budget={a.budget_gib:.2f} requests={n} batch={a.batch} "
      f"ttft_s={res['ttft_s']:.4f} tpot_ms={res['tpot_ms']:.3f} request_s={res['request_s']:.4f} "
      f"footprint_gib={res['footprint_gib']:.2f} peak_gib={_mw.peak_gib():.2f} compute=measured")
