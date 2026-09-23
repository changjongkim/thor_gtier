#!/usr/bin/env python3
"""Run a command and account for the bytes it actually pulled off the device.

Every system compared here delivers weights from storage; what differs is how.
read_bytes from /proc/<pid>/io counts what reached the block layer, so a page
cache hit does not inflate it, which makes it the one number that means the
same thing for a trace driver, an inference engine and a llama.cpp fork.

The counter is polled rather than read once at exit, because /proc/<pid>/io
disappears with the process.  Children are followed too: FlexGen and PowerInfer
are single-process, but an engine that forks a loader would otherwise hide its
reads.
"""
import argparse, json, os, subprocess, sys, threading, time

def read_io(pid):
    try:
        d = {}
        for line in open(f"/proc/{pid}/io"):
            k, _, v = line.partition(":")
            d[k.strip()] = int(v)
        return d.get("read_bytes", 0)
    except (OSError, ValueError):
        return 0

def descendants(pid):
    out = [pid]
    try:
        kids = open(f"/proc/{pid}/task/{pid}/children").read().split()
        for k in kids:
            out += descendants(int(k))
    except (OSError, ValueError):
        pass
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--out", default="")
ap.add_argument("--drop-caches", action="store_true")
ap.add_argument("cmd", nargs=argparse.REMAINDER)
a = ap.parse_args()
cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd

if a.drop_caches:
    subprocess.run("sync", shell=True)
    subprocess.run("echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null",
                   shell=True)

t0 = time.time()
p = subprocess.Popen(cmd)
peak = {"rd": 0}
stop = threading.Event()

def poll():
    # Sum over the tree each tick and keep the maximum: a child that exits
    # early would otherwise take its bytes out of the running total.
    while not stop.is_set():
        tot = sum(read_io(q) for q in descendants(p.pid))
        if tot > peak["rd"]:
            peak["rd"] = tot
        stop.wait(0.05)

th = threading.Thread(target=poll, daemon=True)
th.start()
rc = p.wait()
stop.set(); th.join(timeout=1)
dt = time.time() - t0

res = {"label": a.label, "returncode": rc, "wall_s": round(dt, 3),
       "read_GiB": round(peak["rd"] / 2**30, 3),
       "read_GiB_s": round(peak["rd"] / 2**30 / dt, 4) if dt else 0,
       "cmd": cmd}
print("IOMETER " + json.dumps(res))
if a.out:
    open(a.out, "w").write(json.dumps(res, indent=2))
sys.exit(rc)
