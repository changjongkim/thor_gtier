#!/usr/bin/env python3
"""Fiddler (Kamahori et al., ICLR'25; github.com/efeslab/fiddler) on the same
prompts, timed per request.  Its code is run unmodified except for one knob:
the number of experts it copies to the GPU, which the original derives from
free GPU memory.  On a unified pool that is the whole pool, so the original
would duplicate every expert; here it is set from the budget left after the
model, which Fiddler keeps whole in host memory by design.

usage: fiddler_serve.py --checkpoint DIR --workload W.json --budget-gib B --out O.json
"""
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--workload", required=True)
ap.add_argument("--budget-gib", type=float, required=True)
ap.add_argument("--max-prompt", type=int, default=8192)
ap.add_argument("--max-new", type=int, default=32)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, "/home/thor/kcj/fiddler/src/fiddler")
sys.path.insert(0, "/home/thor/kcj/thor_gtier/scripts")
from memwatch import MemWatch
_mw = MemWatch()
import torch
import transformers
# transformers 4.36 (Fiddler's pin) ships a tokenizers that cannot parse this
# checkpoint's newer tokenizer.json; the SentencePiece model gives the same ids
_from = transformers.AutoTokenizer.from_pretrained
transformers.AutoTokenizer.from_pretrained = lambda *a, **k: _from(*a, **{**k, "use_fast": False})
import mixtral as fid

model_gib = sum(os.path.getsize(os.path.join(a.checkpoint, f)) for f in os.listdir(a.checkpoint)
                if f.endswith(".safetensors")) / 2**30

def budget_n_expert_on_gpu(self):
    n_param = sum(p.numel() for p in self.model.layers[0].block_sparse_moe.experts[0].parameters())
    spare = max(0.0, a.budget_gib - model_gib) * 2**30
    return int(spare // (n_param * 2))
fid.FiddlerMixtral.calc_n_expert_on_gpu = budget_n_expert_on_gpu

class Args: pass
args = Args(); args.model = a.checkpoint; args.cpu_offload = 1; args.beam_width = 1
t0 = time.time()
m = fid.FiddlerMixtral(args)
load_s = time.time() - t0
work = json.load(open(a.workload))
if a.limit: work = work[:a.limit]
rows = []
for w in work:
    ids = m.tokenizer(w["prompt"]).input_ids[:a.max_prompt]
    text = m.tokenizer.decode(ids, skip_special_tokens=True)
    new = min(a.max_new, int(w.get("max_new", a.max_new)))
    ts = time.time()
    pre, dec, hit = m.generate(text, output_token=new)
    te = time.time()
    rows.append({"name": w["name"], "prompt_tok": len(ids), "new_tok": new, "ttft_s": pre,
                 "tpot_ms": dec / max(new - 1, 1) * 1e3, "request_s": te - ts, "gpu_hit_rate": hit})
    print("REQ " + json.dumps(rows[-1]), flush=True)
n = len(rows)
res = {"system": "fiddler", "budget_gib": a.budget_gib, "load_s": load_s, "requests": n,
       "ttft_s": sum(r["ttft_s"] for r in rows) / n, "tpot_ms": sum(r["tpot_ms"] for r in rows) / n,
       "request_s": sum(r["request_s"] for r in rows) / n, "peak_gib": _mw.peak_gib(), "rows": rows}
json.dump(res, open(a.out, "w"), indent=1)
print(f"RESULT policy=fiddler budget={a.budget_gib:.2f} requests={n} ttft_s={res['ttft_s']:.4f} "
      f"tpot_ms={res['tpot_ms']:.3f} request_s={res['request_s']:.4f} peak_gib={res['peak_gib']:.2f} compute=measured")
