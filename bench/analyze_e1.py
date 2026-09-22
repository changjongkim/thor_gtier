#!/usr/bin/env python3
"""Summarise the -ngl sweep: does moving layers between CPU and GPU move bytes?"""
import json, sys
from pathlib import Path

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def table(rows, mmap):
    rs = sorted([r for r in rows if r["mmap"] == mmap], key=lambda r: r["ngl"])
    out = []
    for r in rs:
        out.append((r["ngl"], r.get("tg_tps"), r.get("pp_tps"),
                    r["disk_read_GiB"], r["major_faults"], r["wall_s"]))
    return out

def span(vals):
    vals = [v for v in vals if v is not None]
    return (min(vals), max(vals), (max(vals) / min(vals) - 1) * 100 if min(vals) else 0)

for path in sys.argv[1:]:
    rows = load(path)
    name = rows[0]["model"]
    gib = rows[0]["model_bytes"] / 2**30
    print(f"\n## {name} — {gib:.1f} GiB ({rows[0]['oversub']:.2f}x DRAM)\n")
    for mmap in (1, 0):
        t = table(rows, bool(mmap))
        if not t:
            continue
        print(f"### mmap={'on' if mmap else 'off'}\n")
        print("| -ngl | tg t/s | pp t/s | disk read GiB | major faults | wall s |")
        print("|---:|---:|---:|---:|---:|---:|")
        for ngl, tg, pp, rd, mf, w in t:
            print(f"| {ngl} | {tg} | {pp} | {rd} | {mf} | {w} |")
        lo, hi, pct = span([x[1] for x in t])
        rlo, rhi, rpct = span([x[3] for x in t])
        flo, fhi, fpct = span([float(x[4]) for x in t])
        print(f"\n**throughput {lo}→{hi} t/s ({hi/lo:.2f}x, +{pct:.0f}%)** · "
              f"disk read {rlo}→{rhi} GiB (**{rpct:+.1f}%**) · "
              f"major faults {int(flo)}→{int(fhi)} ({fpct:+.0f}%)\n")
