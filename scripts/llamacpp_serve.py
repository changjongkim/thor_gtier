#!/usr/bin/env python3
"""llama.cpp (github.com/ggml-org/llama.cpp) as the practical baseline: a bf16
GGUF served by llama-server with the experts left memory-mapped from the SSD and
computed on the CPU (--cpu-moe); the rest of the model on the GPU.  The OS page
cache is its expert cache.

Equal memory.  The page cache is this system's cache, so it is not scrubbed; it
is bounded instead by the run's cgroup: --mem-cap-gib sets memory.max of the
cgroup in_cgroup.sh put this process in (anon + page cache).  The measured peak
is max over time of (MemAvailable drop since start + page cache charged to the
cgroup): the drop counts anon memory and cudaMalloc, which MemAvailable sees,
the second term the cached expert pages, which MemAvailable counts as free.
memcal.py calibrates --mem-cap-gib so that this peak equals PHASOR's.

Same prompts: the prompt is tokenized with the checkpoint's HF tokenizer and
sent as token ids (truncated like every runner), greedy, max_new tokens with EOS
ignored (min_new_tokens as the other runners).  TTFT and TPOT are measured at
the client from the streamed tokens.

usage: llamacpp_serve.py --server BIN --gguf F.gguf --tokenizer CKPT --workload W.json
                         --mem-cap-gib B --out O.json [--batch N] [--limit N]
"""
import argparse, json, os, subprocess, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--server", required=True)
ap.add_argument("--gguf", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--workload", required=True)
ap.add_argument("--mem-cap-gib", type=float, default=0.0, help="cgroup memory.max (anon + page cache); 0 = leave")
ap.add_argument("--budget-gib", type=float, default=None, help="alias of --mem-cap-gib (memcal's knob)")
ap.add_argument("--max-prompt", type=int, default=8192)
ap.add_argument("--max-new", type=int, default=32)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--threads", type=int, default=os.cpu_count())
ap.add_argument("--port", type=int, default=18080)
ap.add_argument("--out", required=True)
a = ap.parse_args()
if a.budget_gib is not None: a.mem_cap_gib = a.budget_gib

GIB = 1 << 30
def avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"): return int(l.split()[1]) * 1024
    return 0
def cgroup_dir():
    for l in open("/proc/self/cgroup"):
        if l.startswith("0::"): return "/sys/fs/cgroup" + l.strip()[3:]
    return None
CG = cgroup_dir()
def cg_file():
    try:
        for l in open(f"{CG}/memory.stat"):
            if l.startswith("file "): return int(l.split()[1])
    except Exception: pass
    return 0

if a.mem_cap_gib > 0 and CG and "ledger_bench" in CG:   # only lowers the cap in_cgroup.sh set
    cur = open(f"{CG}/memory.max").read().strip()
    want = int(a.mem_cap_gib * GIB)
    if cur == "max" or want < int(cur):
        subprocess.run(["sudo", "-n", "tee", f"{CG}/memory.max"], input=str(want).encode(),
                       stdout=subprocess.DEVNULL, check=True)

# every run starts cold, as run() drops the page cache for every system: a probe that
# follows another (memcal) would otherwise find the GGUF pages already cached, charged to
# the previous run's (removed) cgroup, and its measured footprint would miss them
# (09-29 01:30: 65% probes read 1.7 GiB after a 42 GiB one)
import glob as _glob
for _p in _glob.glob(os.path.join(os.path.dirname(os.path.abspath(a.gguf)), "*.gguf")):
    try:
        _fd = os.open(_p, os.O_RDONLY); os.posix_fadvise(_fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(_fd)
    except OSError: pass
a0 = avail(); peak = {"total": 0, "drop": 0, "file": 0}; stop = False
def sampler():
    while not stop:
        d, f = a0 - avail(), cg_file()
        peak["total"] = max(peak["total"], d + f); peak["drop"] = max(peak["drop"], d); peak["file"] = max(peak["file"], f)
        time.sleep(0.2)
threading.Thread(target=sampler, daemon=True).start()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.tokenizer)
work = json.load(open(a.workload))
if a.limit: work = work[:a.limit]
ctx = a.batch * (a.max_prompt + a.max_new + 64)
# --load-mode mmap: its "auto" turns mmap off when a device is an iGPU (ggml-cuda: mmap_support =
# type != IGPU, llama.cpp #28160), and then reads the whole model into anonymous memory -- 87 GiB for
# Mixtral, which exhausted the pool (09-28 22:43).  The experts stay on the CPU (--cpu-moe), so the GPU
# never touches the mapped pages; the mapping is what makes the page cache the expert cache.
cmd = [a.server, "-m", a.gguf, "--load-mode", "mmap", "--cpu-moe", "-ngl", "999", "-c", str(ctx), "-np", str(a.batch),
       "-t", str(a.threads), "--host", "127.0.0.1", "--port", str(a.port)]
print("SERVER " + " ".join(cmd), flush=True)
t0 = time.time()
srv = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
URL = f"http://127.0.0.1:{a.port}"
def die(msg, code=1):
    print(msg, file=sys.stderr, flush=True); srv.kill(); os._exit(code)
while True:
    if srv.poll() is not None: die(f"llama-server exited rc={srv.returncode}")
    try:
        if json.load(urllib.request.urlopen(URL + "/health", timeout=2)).get("status") == "ok": break
    except Exception: pass
    if time.time() - t0 > 3600: die("llama-server not ready in 1 h")
    time.sleep(1)
load_s = time.time() - t0


def one(ids, new):
    """one streamed completion: (ttft_s, tpot_ms, request_s, out_ids)"""
    body = json.dumps({"prompt": ids, "n_predict": new, "temperature": 0.0, "top_k": 1, "ignore_eos": True,
                       "cache_prompt": False, "stream": True, "return_tokens": True, "samplers": ["top_k"]}).encode()
    req = urllib.request.Request(URL + "/completion", data=body, headers={"Content-Type": "application/json"})
    ts = time.time(); times, out = [], []
    with urllib.request.urlopen(req, timeout=21600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"): continue
            ev = json.loads(line[5:])
            toks = ev.get("tokens") or ([-1] if ev.get("content") and not ev.get("stop") else [])
            if toks:
                out += toks; times += [time.time()] * len(toks)
            if ev.get("stop"): break
    te = time.time()
    return ((times[0] - ts) if times else te - ts,
            ((times[-1] - times[0]) / (len(times) - 1) * 1e3) if len(times) > 1 else 0.0, te - ts, out)


rows = []
items = []
for w in work:
    ids = tok(w["prompt"]).input_ids[:a.max_prompt]
    items.append((w["name"], ids, min(a.max_new, int(w.get("max_new", a.max_new)))))
if a.batch == 1:
    for name, ids, new in items:
        ttft, tpot, rs, out = one(ids, new)
        rows.append({"name": name, "prompt_tok": len(ids), "new_tok": len(out), "ttft_s": ttft, "tpot_ms": tpot,
                     "request_s": rs, "out_ids": out})
        print("REQ " + json.dumps({k: v for k, v in rows[-1].items() if k != "out_ids"}), flush=True)
else:   # E3: groups of `batch` requests sent at once (the server's -np slots decode them together)
    for g in range(0, len(items), a.batch):
        grp = items[g:g + a.batch]; res = [None] * len(grp)
        def go(i, it): res[i] = one(it[1], it[2])
        th = [threading.Thread(target=go, args=(i, it)) for i, it in enumerate(grp)]
        ts = time.time(); [t.start() for t in th]; [t.join() for t in th]; te = time.time()
        rows.append({"names": [it[0] for it in grp], "batch": len(grp), "prompt_tok": [len(it[1]) for it in grp],
                     "new_tok": min(len(r[3]) for r in res), "ttft_s": max(r[0] for r in res),
                     "tpot_ms": max(r[1] for r in res), "request_s": te - ts})
        rows[-1]["tok_per_s"] = sum(len(r[3]) for r in res) / (te - ts)
        print("REQ " + json.dumps(rows[-1]), flush=True)
stop = True; time.sleep(0.3)
srv.terminate()
try: srv.wait(timeout=30)
except Exception: srv.kill()
n = len(rows)
res = {"system": "llamacpp", "budget_gib": a.mem_cap_gib, "load_s": load_s, "requests": n, "batch": a.batch,
       "ttft_s": sum(r["ttft_s"] for r in rows) / n, "tpot_ms": sum(r["tpot_ms"] for r in rows) / n,
       "request_s": sum(r["request_s"] for r in rows) / n, "peak_gib": peak["total"] / GIB,
       "peak_drop_gib": peak["drop"] / GIB, "peak_cgroup_file_gib": peak["file"] / GIB, "rows": rows}
if a.batch > 1:
    res["tok_per_s"] = sum(r["batch"] * r["new_tok"] for r in rows) / sum(r["request_s"] for r in rows)
json.dump(res, open(a.out, "w"), indent=1)
print(f"RESULT policy=llamacpp budget={a.mem_cap_gib:.2f} requests={n} ttft_s={res['ttft_s']:.4f} "
      f"tpot_ms={res['tpot_ms']:.3f} request_s={res['request_s']:.4f} peak_gib={res['peak_gib']:.2f} compute=measured", flush=True)
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
