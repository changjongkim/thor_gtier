#!/usr/bin/env python3
"""Measure out-of-core LLM inference on a coherent edge SoC.

For each (model, -ngl, mmap) configuration this records not just throughput but
the storage and paging behaviour underneath it: major page faults, NVMe sectors
read, page-cache growth and board power.  Those are what distinguish a tier that
works from one that thrashes.

The -ngl sweep is the load-bearing experiment.  On a discrete GPU, -ngl K decides
how many layers live in VRAM and how many are streamed from host memory per
token.  On Thor, CUDA reports the whole 122 GiB of system memory as VRAM, so the
GPU/host split that FlexGen, PowerInfer, ZeRO-Infinity, FlashNeuron, DeepUM and
G10 all build on has no physical meaning.  If data movement is flat across -ngl,
the three-tier premise is falsified on this hardware.
"""

import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

LLAMA = Path("/home/thor/skim/llama.cpp/build/bin/llama-bench")


def vmstat():
    d = {}
    for line in open("/proc/vmstat"):
        k, v = line.split()
        d[k] = int(v)
    return d


def diskstats(dev="nvme0n1"):
    for line in open("/proc/diskstats"):
        f = line.split()
        if len(f) > 13 and f[2] == dev:
            # fields: reads_completed, reads_merged, sectors_read, ms_reading, ...
            return {"sectors_read": int(f[5]), "sectors_written": int(f[9])}
    return {"sectors_read": 0, "sectors_written": 0}


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, _, v = line.partition(":")
        d[k] = int(v.split()[0]) * 1024
    return d


def power_w():
    """Board input power from tegrastats, in watts."""
    try:
        p = subprocess.run(["timeout", "3", "tegrastats", "--interval", "500"],
                           capture_output=True, text=True)
        m = re.findall(r"VIN (\d+)mW", p.stdout)
        return max(int(x) for x in m) / 1000.0 if m else None
    except Exception:
        return None


def drop_caches():
    subprocess.run("sync", shell=True)
    subprocess.run("echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1",
                   shell=True)
    time.sleep(2)


def run_one(model, ngl, mmap, pp, tg, reps, timeout_s):
    cmd = [str(LLAMA), "-m", model, "-ngl", str(ngl),
           "-p", str(pp), "-n", str(tg), "-r", str(reps), "-o", "json"]
    if not mmap:
        cmd += ["--mmap", "0"]

    drop_caches()
    v0, d0, m0 = vmstat(), diskstats(), meminfo()
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        wall = time.time() - t0
        ok, err = p.returncode == 0, p.stderr[-800:]
    except subprocess.TimeoutExpired:
        wall, ok, err, p = time.time() - t0, False, "TIMEOUT", None
    v1, d1, m1 = vmstat(), diskstats(), meminfo()

    res = {
        "model": os.path.basename(os.path.dirname(model)) or os.path.basename(model),
        "model_bytes": os.path.getsize(model) if os.path.exists(model) else None,
        "ngl": ngl, "mmap": mmap, "pp": pp, "tg": tg,
        "ok": ok, "wall_s": round(wall, 2),
        "major_faults": v1["pgmajfault"] - v0["pgmajfault"],
        "minor_faults": v1["pgfault"] - v0["pgfault"],
        "disk_read_GiB": round((d1["sectors_read"] - d0["sectors_read"]) * 512 / 2**30, 3),
        "disk_write_GiB": round((d1["sectors_written"] - d0["sectors_written"]) * 512 / 2**30, 3),
        "cache_growth_GiB": round((m1["Cached"] - m0["Cached"]) / 2**30, 2),
        "mem_avail_after_GiB": round(m1["MemAvailable"] / 2**30, 1),
        "error": None if ok else err,
    }
    if ok and p:
        try:
            for row in json.loads(p.stdout):
                res[row["n_prompt"] and "pp_tps" or "tg_tps"] = round(row["avg_ts"], 2)
        except Exception as e:
            res["parse_error"] = str(e)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--ngl", nargs="+", type=int, default=[0, 99])
    ap.add_argument("--mmap", nargs="+", type=int, default=[1, 0])
    ap.add_argument("--pp", type=int, default=128)
    ap.add_argument("--tg", type=int, default=32)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default="results/bench.jsonl")
    a = ap.parse_args()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    dram = meminfo()["MemTotal"] / 2**30
    print(f"DRAM {dram:.1f} GiB\n")
    with open(a.out, "a") as fh:
        for model in a.models:
            sz = os.path.getsize(model) / 2**30 if os.path.exists(model) else 0
            for mm in a.mmap:
                for ngl in a.ngl:
                    print(f"  {os.path.basename(model)[:42]:42s} "
                          f"{sz:6.1f}GiB ({sz/dram:.2f}x) ngl={ngl:<3d} mmap={mm} ... ",
                          end="", flush=True)
                    r = run_one(model, ngl, bool(mm), a.pp, a.tg, a.reps, a.timeout)
                    r["oversub"] = round(sz / dram, 3)
                    fh.write(json.dumps(r) + "\n"); fh.flush()
                    print(f"{'OK ' if r['ok'] else 'FAIL'} "
                          f"tg={r.get('tg_tps','-')} t/s  majflt={r['major_faults']:,} "
                          f"read={r['disk_read_GiB']}GiB  {r['wall_s']}s")


if __name__ == "__main__":
    main()
