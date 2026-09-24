#!/usr/bin/env python3
"""Record the experts a model actually routes to, over a workload file.

Replaces the hand-written prompt list in capture_routing.py: the same
workload can now be run against every model, which is what makes the
per-model comparison a comparison and not a collection.

The router's Linear is hooked and the same top-k the model takes is read off
its output, so nothing depends on internals beyond that module existing --
which is what lets one script serve Qwen3-MoE's 128-of-8 and Mixtral's
8-of-2 without knowing anything about either.
"""
import argparse, json, os, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--workload", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-prompt-tokens", type=int, default=8192)
ap.add_argument("--decode", type=int, default=0, help="0 = use each prompt's max_new")
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()

rows = json.load(open(a.workload))
tok = AutoTokenizer.from_pretrained(a.model)
t0 = time.time()
# sdpa rather than eager: the hooks sit on mlp.gate and do not need the
# attention weights materialised, and eager keeps an n-by-n matrix per head
# per layer -- at 8192 tokens that is several gigabytes a layer, which is
# what killed the long-prompt captures.
model = AutoModelForCausalLM.from_pretrained(
    a.model, dtype=torch.bfloat16, device_map=a.device,
    attn_implementation="sdpa", low_cpu_mem_usage=True)
model.eval()
cfg = model.config
TOPK = getattr(cfg, "num_experts_per_tok", None)
NEXP = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None)
L = cfg.num_hidden_layers
print(f"loaded in {time.time()-t0:.1f}s | layers={L} experts={NEXP} topk={TOPK}", flush=True)
if not TOPK or not NEXP:
    raise SystemExit("not a routed MoE checkpoint")

records, state = [], {"tag": "", "base": 0}
def mk_hook(layer):
    def hook(_m, _i, out):
        # Recent transformers route through a TopKRouter module that returns
        # (router_logits, router_scores, router_indices).  The indices are the
        # experts the model actually uses, so they are read directly; only a
        # router that returns bare logits falls back to taking the top-k
        # here, which picks the same experts since softmax is monotone.
        if isinstance(out, (tuple, list)) and len(out) >= 3 and \
           isinstance(out[2], torch.Tensor) and out[2].dtype in (torch.int64, torch.int32):
            idx = out[2].reshape(-1, out[2].shape[-1]).cpu().numpy()
        else:
            logits = out if isinstance(out, torch.Tensor) else out[0]
            logits = logits.reshape(-1, logits.shape[-1]).float()
            idx = torch.topk(logits, TOPK, dim=-1).indices.cpu().numpy()
        for t in range(idx.shape[0]):
            records.append((state["tag"], layer, state["base"] + t, idx[t]))
    return hook

hooks = []
for i, layer in enumerate(model.model.layers):
    mlp = getattr(layer, "mlp", None)
    gate = getattr(mlp, "gate", None) if mlp is not None else None
    # Any module named gate under the MoE block: a plain Linear in older
    # transformers, a TopKRouter in newer ones.  Requiring nn.Linear hooked
    # nothing on the installed version and every capture came back empty.
    if gate is not None and isinstance(gate, torch.nn.Module):
        hooks.append(gate.register_forward_hook(mk_hook(i)))
print(f"hooked {len(hooks)} routers", flush=True)
if not hooks:
    raise SystemExit("no router modules found -- refusing to write an empty trace")

for r in rows:
    ids = tok.encode(r["prompt"], return_tensors="pt")
    if ids.shape[1] > a.max_prompt_tokens:
        ids = ids[:, :a.max_prompt_tokens]
    ids = ids.to(a.device)
    nd = a.decode or int(r.get("max_new", 16))
    state["tag"], state["base"] = r["name"] + "/prefill", 0
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    del out
    state["tag"] = r["name"] + "/decode"
    for s in range(nd):
        state["base"] = s
        with torch.no_grad():
            o = model(nxt, past_key_values=past, use_cache=True)
        past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
    del past, nxt
    torch.cuda.empty_cache()
    print(f"  {r['name']}: {ids.shape[1]} prompt tok, +{nd}", flush=True)

for h in hooks: h.remove()

tags = sorted({r[0] for r in records})
tid = {t: i for i, t in enumerate(tags)}
np.savez_compressed(
    a.out,
    tag=np.array([tid[r[0]] for r in records], dtype=np.int16),
    layer=np.array([r[1] for r in records], dtype=np.int16),
    pos=np.array([r[2] for r in records], dtype=np.int32),
    expert=np.stack([r[3] for r in records]).astype(np.int16),
    tags=np.array(tags), n_experts=NEXP, topk=TOPK, n_layers=L,
    family=np.array([next((x["family"] for x in rows if x["name"] == t.split("/")[0]), "")
                     for t in tags]))
print(f"wrote {a.out}: {len(records)} routing decisions over {len(rows)} prompts", flush=True)
