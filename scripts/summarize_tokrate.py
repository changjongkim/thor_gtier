#!/usr/bin/env python3
"""The policy comparison on the axis an engine is judged by."""
import glob, os, re
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "TOKRATE")
RE_H = re.compile(r"residency\s+([\d.]+) GiB \(([\d.]+)% of experts\)\s+policy=(\S+)")
RE_R = re.compile(r"io\s+([\d.]+) ms/tok \| compute\s+([\d.]+) ms \(floor\s+([\d.]+)\) "
                  r"-> \*\*\s*([\d.]+) tok/s\*\* \(ceiling\s+([\d.]+)\) \| prefill I/O\s+([\d.]+) s")
def parse(p):
    d={"name":os.path.basename(p)[:-4]}
    for line in open(p, errors="ignore"):
        m=RE_H.search(line)
        if m: d.update(resident=float(m.group(1)),pct=float(m.group(2)),policy=m.group(3))
        m=RE_R.search(line)
        if m: d.update(io=float(m.group(1)),cp=float(m.group(2)),floor=float(m.group(3)),
                       tps=float(m.group(4)),ceil=float(m.group(5)),ttft=float(m.group(6)))
    return d if "tps" in d else None
runs={}
for p in sorted(glob.glob(os.path.join(OUT,"t_*.txt"))):
    r=parse(p)
    if r: runs[r["name"]]=r
CAT={"lru":"기본","moe-inf*":"SOTA 재구현","mixtral*":"SOTA 재구현"}
ORDER=[1,10,11,2,3,4,7,5,6,8,9]

print("# 실제 토큰 레이트 — 정책 비교\n")
print("Qwen3-30B-A3B, 실측 라우팅, 실행마다 캐시 비움. 라우팅된 전문가의 feed-forward를 "
      "**실제로 실행**하므로 전송 초가 아니라 토큰 레이트다.\n")
print("연산 커널은 모든 정책에서 동일하다 — 행을 가르는 것은 각 정책이 일으키는 I/O다. "
      "커널이 튜닝된 것이 아니므로 측정값과 메모리 바운드 하한(토큰당 읽는 바이트 / 254 GB/s, "
      "실측 GPU 읽기 대역폭)에서의 값을 함께 싣는다. 진실은 그 사이에 있다.\n")
for B in (24,40):
    rows=[runs[f"t_b{B}_p{P}"] for P in ORDER if f"t_b{B}_p{P}" in runs]
    if not rows: continue
    print(f"## 예산 {B} GiB — 상주 {rows[0]['resident']:.1f} GiB ({rows[0]['pct']:.0f}% of experts)\n")
    print("| 정책 | 분류 | I/O ms/tok | 연산 ms/tok | **tok/s** | 하한에서 tok/s | 프리필 I/O s |")
    print("|---|---|---:|---:|---:|---:|---:|")
    best=max(r["tps"] for r in rows)
    for r in rows:
        cat=CAT.get(r["policy"],"본 연구")
        t=f"**{r['tps']:.3f}**" if r["tps"]==best else f"{r['tps']:.3f}"
        print(f"| `{r['policy']}` | {cat} | {r['io']:.2f} | {r['cp']:.2f} | {t} | "
              f"{r['ceil']:.3f} | {r['ttft']:.3f} |")
    u=next((r for r in rows if r["policy"]=="unified"),None)
    sota=[r for r in rows if CAT.get(r["policy"]) in ("기본","SOTA 재구현")]
    if u and sota:
        b=max(sota,key=lambda r:r["tps"]); w=min(sota,key=lambda r:r["tps"])
        print(f"\n`unified` 대 최강 베이스라인 `{b['policy']}`: **{u['tps']/b['tps']:.2f}배**, "
              f"최약 `{w['policy']}` 대비 **{u['tps']/w['tps']:.2f}배**.\n")
