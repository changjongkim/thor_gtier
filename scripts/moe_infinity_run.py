#!/usr/bin/env python3
"""Run MoE-Infinity on a local checkpoint and account for the bytes it reads.

The comparable quantity across every system here is weight delivery: how many
bytes of model weights reach the GPU per second from cold storage.  An engine
does routing, attention and matmul on top of that, so tok/s alone cannot be
lined up against a trace driver.  What can be lined up is read_bytes over
decode wall time, which is what this records.
"""
import argparse, json, os, sys, time

def io_counters():
    # read_bytes counts what actually went to the block layer, so page-cache
    # hits do not inflate it -- exactly the number the trace driver reports.
    d = {}
    try:
        for line in open("/proc/self/io"):
            k, _, v = line.partition(":")
            d[k.strip()] = int(v)
    except OSError:
        pass
    return d

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--offload_dir", required=True)
ap.add_argument("--max_new_tokens", type=int, default=32)
ap.add_argument("--device_memory_ratio", type=float, default=0.75)
ap.add_argument("--out", default="")
a = ap.parse_args()

import torch
from transformers import AutoTokenizer
from moe_infinity import MoE

tok = AutoTokenizer.from_pretrained(a.checkpoint)
t_load0 = time.time()
io_load0 = io_counters()
model = MoE(a.checkpoint, {"offload_path": a.offload_dir,
                           "device_memory_ratio": a.device_memory_ratio})
load_s = time.time() - t_load0
io_load1 = io_counters()

prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Explain why sorting is O(n log n)."}],
    tokenize=False, add_generation_prompt=True)
ids = tok.encode(prompt, return_tensors="pt").to("cuda:0")

# One short warm generate first: the interesting number is steady-state decode,
# not the one-off cost of faulting in whatever the engine keeps resident.
with torch.no_grad():
    model.generate(ids, max_new_tokens=2, do_sample=False,
                   pad_token_id=tok.eos_token_id)

torch.cuda.synchronize()
io0 = io_counters(); t0 = time.time()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=a.max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()
dt = time.time() - t0; io1 = io_counters()

new_tok = out.shape[1] - ids.shape[1]
rd = io1.get("read_bytes", 0) - io0.get("read_bytes", 0)
res = {
    "system": "MoE-Infinity",
    "checkpoint": a.checkpoint,
    "load_s": round(load_s, 2),
    "load_read_GiB": round((io_load1.get("read_bytes", 0)
                            - io_load0.get("read_bytes", 0)) / 2**30, 3),
    "tokens": new_tok,
    "decode_s": round(dt, 3),
    "tok_s": round(new_tok / dt, 4) if dt else 0,
    "decode_read_GiB": round(rd / 2**30, 3),
    "deliver_GiB_s": round(rd / 2**30 / dt, 4) if dt else 0,
    "device_memory_ratio": a.device_memory_ratio,
}
print(json.dumps(res, indent=2))
print(tok.decode(out[0], skip_special_tokens=True)[:200])
if a.out:
    open(a.out, "w").write(json.dumps(res, indent=2))
