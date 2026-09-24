#!/usr/bin/env python3
"""Turn the three datasets into one prompt format, so every model sees the
same requests.

Each carries a different claim.

  longbench  prompts of thousands of tokens.  The routing capture so far
             topped out at 613, where the prefill union reaches 79% of the
             experts; the claim that prefill dominates has to survive prompts
             an order of magnitude longer or be withdrawn.

  sharegpt   multi-turn conversations.  Turn k of a conversation repeats
             turns 1..k-1 verbatim, so prefix reuse arises from the workload
             instead of being staged with hand-written system prompts.

  mmlu       five-shot questions across 57 subjects.  The few-shot block is a
             shared prefix per subject, and the subjects give the domain
             diversity the single-domain trace lacks.

Output: a JSON list of {name, family, prompt, max_new}, where `family` marks
requests that share a prefix -- which the driver verifies from routing rather
than trusting (sec 3.5).
"""
import argparse, glob, json, os, random, re

D = "/home/thor/kcj/datasets"

def longbench(n, seed, max_chars):
    out, rng = [], random.Random(seed)
    files = sorted(glob.glob(f"{D}/longbench/data/*.jsonl"))
    files = [f for f in files if not f.endswith("_e.jsonl")]
    rng.shuffle(files)
    per = max(1, n // max(1, len(files)))
    for f in files:
        task = os.path.basename(f)[:-6]
        rows = []
        with open(f) as fh:
            for line in fh:
                try: rows.append(json.loads(line))
                except Exception: pass
        rng.shuffle(rows)
        for r in rows[:per]:
            ctx = (r.get("context") or "")[:max_chars]
            q = r.get("input") or ""
            if not ctx: continue
            out.append({"name": f"lb_{task}_{len(out)}", "family": f"lb_{task}",
                        "prompt": f"{ctx}\n\nQuestion: {q}\nAnswer:", "max_new": 32})
            if len(out) >= n: return out
    return out

def sharegpt(n, seed):
    """One request per turn, so a conversation's later turns repeat its
    earlier ones -- the prefix reuse a real session produces."""
    out, rng = [], random.Random(seed)
    p = f"{D}/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
    convs = json.load(open(p))
    rng.shuffle(convs)
    for c in convs:
        turns = [t for t in c.get("conversations", []) if t.get("value")]
        if len(turns) < 4: continue
        cid = c.get("id", str(len(out)))[:12]
        hist = ""
        used = 0
        for t in turns[:6]:
            role = "User" if t.get("from") in ("human", "user") else "Assistant"
            hist += f"{role}: {t['value'].strip()}\n"
            if role != "User": continue
            used += 1
            if used < 2: continue           # first turn has no shared prefix yet
            out.append({"name": f"sg_{cid}_t{used}", "family": f"sg_{cid}",
                        "prompt": hist + "Assistant:", "max_new": 32})
            if len(out) >= n: return out
    return out

def mmlu(n, seed, shots=5, n_subjects=6):
    """Five-shot per subject: the shot block is identical across that
    subject's questions, which is a shared prefix by construction."""
    out, rng = [], random.Random(seed)
    subjects = sorted(d for d in os.listdir(f"{D}/mmlu")
                      if os.path.isdir(f"{D}/mmlu/{d}") and d not in ("all","auxiliary_train"))
    rng.shuffle(subjects)
    # Few subjects with several questions each, so the shot block is actually
    # shared: one question per subject would give 24 families of one and no
    # prefix reuse to measure.
    subjects = subjects[:n_subjects]
    per = max(2, n // max(1, len(subjects)))
    for sub in subjects:
        rows = []
        for f in sorted(glob.glob(f"{D}/mmlu/{sub}/*.parquet")):
            try:
                import pyarrow.parquet as pq
                t = pq.read_table(f).to_pylist()
                rows.extend(t)
            except Exception:
                pass
        if len(rows) < shots + 1: continue
        rng.shuffle(rows)
        def fmt(r, with_ans=True):
            ch = r.get("choices") or []
            body = f"Question: {r.get('question','')}\n"
            for i, c in enumerate(ch):
                body += f"{chr(65+i)}. {c}\n"
            if with_ans:
                a = r.get("answer")
                body += f"Answer: {chr(65+a) if isinstance(a,int) else a}\n\n"
            else:
                body += "Answer:"
            return body
        shot_block = f"The following are multiple choice questions about {sub.replace('_',' ')}.\n\n"
        shot_block += "".join(fmt(r) for r in rows[:shots])
        for r in rows[shots:shots+per]:
            out.append({"name": f"mmlu_{sub}_{len(out)}", "family": f"mmlu_{sub}",
                        "prompt": shot_block + fmt(r, False), "max_new": 8})
            if len(out) >= n: return out
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--workload", required=True, choices=["longbench","sharegpt","mmlu"])
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--seed", type=int, default=20260924)
ap.add_argument("--max-chars", type=int, default=60000)
ap.add_argument("--subjects", type=int, default=6)
ap.add_argument("--out", required=True)
a = ap.parse_args()

fn = {"longbench": lambda: longbench(a.n, a.seed, a.max_chars),
      "sharegpt":  lambda: sharegpt(a.n, a.seed),
      "mmlu":      lambda: mmlu(a.n, a.seed, n_subjects=a.subjects)}[a.workload]
rows = fn()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(rows, open(a.out, "w"), ensure_ascii=False)
fams = len({r["family"] for r in rows})
chars = [len(r["prompt"]) for r in rows] or [0]
print(f"{a.workload}: {len(rows)} prompts, {fams} families, "
      f"chars min {min(chars)} median {sorted(chars)[len(chars)//2]} max {max(chars)}")
