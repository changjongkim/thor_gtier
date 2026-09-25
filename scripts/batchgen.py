"""E3: serve a workload in groups of `batch` requests through a runner's own
model.generate, the same way for every system: left padding, greedy, one
max_new per group (the smallest of the group's limits), per-step timestamps
from a logits processor (a streamer only supports batch 1).

Each row is one group: its request latency is the group's wall time, TTFT the
time to its first generated token, TPOT the mean step time after that, and
tok_per_s the group's generated tokens over its wall time."""
import json, time
import torch
from transformers import LogitsProcessorList


class StepClock:
    def __init__(self): self.t = []
    def __call__(self, input_ids, scores):
        torch.cuda.synchronize(); self.t.append(time.time()); return scores


def serve(generate, tok, work, batch, max_prompt, max_new, stats=None):
    """generate(**kwargs) -> output ids; stats() -> dict of counters (optional)."""
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    rows = []
    for g0 in range(0, len(work), batch):
        grp = work[g0:g0 + batch]
        enc = tok([w["prompt"] for w in grp], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_prompt).to("cuda:0")
        new = min(min(max_new, int(w.get("max_new", max_new))) for w in grp)
        s0 = stats() if stats else {}
        clk = StepClock(); torch.cuda.synchronize(); ts = time.time()
        with torch.no_grad():
            generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=new,
                     min_new_tokens=new, do_sample=False, pad_token_id=tok.pad_token_id,
                     logits_processor=LogitsProcessorList([clk]))
        torch.cuda.synchronize(); te = time.time()
        s1 = stats() if stats else {}
        row = {"names": [w["name"] for w in grp], "batch": len(grp),
               "prompt_tok": [int(m.sum()) for m in enc["attention_mask"]], "new_tok": new,
               "ttft_s": (clk.t[0] - ts) if clk.t else te - ts,
               "tpot_ms": ((clk.t[-1] - clk.t[0]) / (len(clk.t) - 1) * 1e3) if len(clk.t) > 1 else 0.0,
               "request_s": te - ts, "tok_per_s": len(grp) * new / (te - ts)}
        row.update({k: s1[k] - s0[k] for k in s1})
        rows.append(row)
        print("REQ " + json.dumps(row), flush=True)
    return rows


def summary(rows):
    return {"tok_per_s": sum(r["batch"] * r["new_tok"] for r in rows) / sum(r["request_s"] for r in rows)}
