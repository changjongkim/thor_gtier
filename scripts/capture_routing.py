#!/usr/bin/env python3
"""Log which experts Qwen3-MoE actually routes to, per layer and per token.

Every claim about residency, caching and batching in this repository is a
function of the routing distribution, and that distribution has so far been
assumed (Zipf, skew 0.8) rather than measured.  Two different things matter
and they are not the same:

  - the marginal popularity of each expert, which is what a static
    popularity-ordered residency can exploit;
  - the temporal structure -- whether a sequence, or two requests sharing a
    prefix, return to the same experts -- which is what a cache exploits.

Hooking the router's Linear and taking the same top-k the model takes gives
both, without depending on internals beyond the gate module's output.
"""
import argparse, json, os, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/thor/kcj/models/qwen3_30b_a3b")
ap.add_argument("--out", default="results/SCOPE/routing.npz")
ap.add_argument("--decode", type=int, default=64)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    a.model, dtype=torch.bfloat16, device_map=a.device, attn_implementation="eager")
model.eval()
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

cfg = model.config
TOPK = cfg.num_experts_per_tok
NEXP = cfg.num_experts
print(f"layers={cfg.num_hidden_layers} experts={NEXP} topk={TOPK}", flush=True)

# (layer, token) -> the topk expert ids the router chose
records = []          # list of (tag, layer, token_index, [ids])
state = {"tag": "", "base": 0}

def mk_hook(layer):
    def hook(_mod, _inp, out):
        # out: [tokens, n_experts] router logits
        logits = out if isinstance(out, torch.Tensor) else out[0]
        logits = logits.reshape(-1, logits.shape[-1]).float()
        idx = torch.topk(logits, TOPK, dim=-1).indices.cpu().numpy()
        for t in range(idx.shape[0]):
            records.append((state["tag"], layer, state["base"] + t, idx[t]))
    return hook

hooks = []
for i, layer in enumerate(model.model.layers):
    gate = getattr(getattr(layer, "mlp", None), "gate", None)
    if gate is not None:
        hooks.append(gate.register_forward_hook(mk_hook(i)))
print(f"hooked {len(hooks)} routers", flush=True)

SHARED = ("You are a careful systems engineer. Answer precisely and briefly. "
          "Consider the following context about storage hardware: NVMe drives "
          "deliver high throughput at large block sizes and collapse at small "
          "ones, and the page cache hides this from most applications. ")
# A long prompt is needed on its own: the simulation predicted the prefill
# union reaches all 128 experts by ~256 tokens, and that only shows up if a
# prompt is actually that long.
LONG = (SHARED + " ") * 12 + (
    "Now write a detailed comparison of io_uring, POSIX AIO and synchronous "
    "pread for streaming model weights from NVMe into GPU-addressable memory, "
    "covering queue depth, alignment, and completion handling. ")

PROMPTS = [
    ("long_prefill", LONG),
    ("distinct_a", "Explain why merge sort is O(n log n) in the worst case."),
    ("distinct_b", "Describe the role of the trans-Golgi network in secretion."),
    ("shared_1", SHARED + "Question: why does random I/O improve with queue depth?"),
    ("shared_2", SHARED + "Question: what limits sequential read on a DRAM-less SSD?"),
    ("shared_3", SHARED + "Question: how does O_DIRECT change the read path?"),
]

for tag, text in PROMPTS:
    msgs = [{"role": "user", "content": text}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(p, return_tensors="pt").to(a.device)
    state["tag"], state["base"] = tag + "/prefill", 0
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    state["tag"] = tag + "/decode"
    for s in range(a.decode):
        state["base"] = s
        with torch.no_grad():
            o = model(nxt, past_key_values=past, use_cache=True)
        past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
    print(f"{tag}: prompt {ids.shape[1]} tok, +{a.decode} decoded", flush=True)

for h in hooks:
    h.remove()

tags = sorted({r[0] for r in records})
tag_id = {t: i for i, t in enumerate(tags)}
arr_tag = np.array([tag_id[r[0]] for r in records], dtype=np.int16)
arr_lay = np.array([r[1] for r in records], dtype=np.int16)
arr_pos = np.array([r[2] for r in records], dtype=np.int32)
arr_exp = np.stack([r[3] for r in records]).astype(np.int16)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
np.savez_compressed(a.out, tag=arr_tag, layer=arr_lay, pos=arr_pos, expert=arr_exp,
                    tags=np.array(tags), n_experts=NEXP, topk=TOPK,
                    n_layers=cfg.num_hidden_layers)
print(f"wrote {a.out}: {len(records)} routing decisions", flush=True)
