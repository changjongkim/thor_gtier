#!/usr/bin/env python3
"""Capture a workload's expert routing from a bf16 checkpoint too large to load
whole, by running it through PHASOR-HF (whose tokens equal stock transformers).
Same record format as scripts/capture_workload.py: prefill once, then greedy
decode, one record per (token, layer) with the router's top-k.

usage: capture_routing.py --checkpoint DIR --workload W.json --out O.npz --budget-gib B
       [--slot-mib 128 --window-gib 1.5] [--max-prompt 8192]
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phasor_hf
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True); ap.add_argument("--workload", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--budget-gib", type=float, required=True)
ap.add_argument("--slot-mib", type=int, default=128); ap.add_argument("--window-gib", type=float, default=1.5)
ap.add_argument("--max-prompt", type=int, default=8192)
a = ap.parse_args()
model, tok, eng = phasor_hf.build(a.checkpoint, a.budget_gib, window_gib=a.window_gib, slot_mib=a.slot_mib)
rows = json.load(open(a.workload))
blocks = [m for m in model.modules() if isinstance(m, phasor_hf.PhasorMoE)]
L, K, E = len(blocks), blocks[0].k, blocks[0].E
records, state = [], {"tag": "", "base": 0}
def mk_hook(layer):
    def hook(_m, _i, out):
        idx = torch.topk(out.float(), K, dim=-1).indices.reshape(-1, K).cpu().numpy()
        for t in range(idx.shape[0]):
            records.append((state["tag"], layer, state["base"] + t, idx[t]))
    return hook
hooks = [b.gate.register_forward_hook(mk_hook(b.layer)) for b in blocks]
for r in rows:
    ids = tok.encode(r["prompt"], return_tensors="pt")[:, :a.max_prompt].cuda()
    nd = int(r.get("max_new", 16))
    eng.begin_request()
    state["tag"], state["base"] = r["name"] + "/prefill", 0
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1); del out
    state["tag"] = r["name"] + "/decode"
    for s in range(nd):
        state["base"] = s
        with torch.no_grad():
            o = model(nxt, past_key_values=past, use_cache=True)
        past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
    del past, nxt; torch.cuda.empty_cache()
    print(f"  {r['name']}: {ids.shape[1]} prompt tok, +{nd}", flush=True)
for h in hooks: h.remove()
tags = sorted({r[0] for r in records}); tid = {t: i for i, t in enumerate(tags)}
np.savez_compressed(a.out,
    tag=np.array([tid[r[0]] for r in records], dtype=np.int16),
    layer=np.array([r[1] for r in records], dtype=np.int16),
    pos=np.array([r[2] for r in records], dtype=np.int32),
    expert=np.array([r[3] for r in records], dtype=np.int16),
    tags=np.array(tags), n_experts=E, topk=K, n_layers=L,
    family=np.array([next((x["family"] for x in rows if x["name"] == t.split("/")[0]), "") for t in tags]))
print(f"wrote {a.out}: {len(records)} records, {len(tags)//2} requests")
