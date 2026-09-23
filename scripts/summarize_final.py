#!/usr/bin/env python3
"""The definitive tables: every policy against every budget, one trace.

Ordered so the reading is: what the published ideas do, what the components
here add, and what putting the two phases on one value scale does to the
tradeoff between them.
"""
import glob, os, re

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "FINAL")
RE_HEAD = re.compile(r"budget\s+(?P<budget>[\d.]+) GiB.*?residency\s+(?P<resident>[\d.]+) GiB "
                     r"\((?P<res_pct>[\d.]+)% of experts\)\s+policy=(?P<policy>\S+)")
RE_RES = re.compile(r"TTFT\(io\)\s+(?P<ttft>[\d.]+) s\s+prefill\s+(?P<pre>[\d.]+) GiB \| "
                    r"look=(?P<look>\d+)\s+\| TPOT\(io\)\s+(?P<tpot>[\d.]+) ms\s+"
                    r"decode\s+(?P<dec>[\d.]+) GiB/tok \| stall\s+(?P<stall>[\d.]+) s")
RE_FAM = re.compile(r"prefix families:(?P<fam>.*)")
F = {"budget","resident","res_pct","ttft","pre","tpot","dec","stall"}

def parse(p):
    d = {"name": os.path.basename(p)[:-4]}
    for line in open(p, errors="ignore"):
        for rx in (RE_HEAD, RE_RES, RE_FAM):
            m = rx.search(line)
            if not m: continue
            for k, v in m.groupdict().items():
                d[k] = float(v) if k in F else v.strip()
    return d if "ttft" in d else None

runs = {}
for p in sorted(glob.glob(os.path.join(OUT, "f_*.txt"))):
    r = parse(p)
    if r: runs[r["name"]] = r

POL = [(1,"lru","기본"),(10,"moe-inf*","SOTA 재구현"),(11,"mixtral*","SOTA 재구현"),
       (2,"lru+phase","본 연구 §3.6"),(3,"per-layer","본 연구 §3.4"),
       (4,"prefix","본 연구 §3.5"),(7,"multi-prefix","본 연구 §3.5"),
       (5,"online","본 연구 §3.4"),(6,"online+prefix","본 연구 §3.4+3.5"),
       (8,"unified","본 연구 §3.7"),(9,"unified-online","본 연구 §3.7")]

print("# 결정판 비교 — 모든 정책, 모든 예산, 하나의 트레이스\n")
print("Qwen3-30B-A3B(56.9 GiB, 전문가 유닛 6144개 x 9.00 MiB), 실측 라우팅 "
      "(시스템 프롬프트 3종 x 3요청 + 613토큰 프롬프트 + 단독 2건), 실행마다 캐시 비움, "
      "상주는 실제로 할당한다.\n")
print("`*`가 붙은 것은 공개된 **아이디어의 재구현**이지 그 시스템 자체가 아니다.\n")

if "f_none" in runs:
    r = runs["f_none"]
    print(f"상주 없음(바닥): TTFT **{r['ttft']:.3f} s**, TPOT **{r['tpot']:.2f} ms**\n")

for B in (16,24,32,40,48):
    rows = [(lab, cat, runs[f"f_b{B}_p{P}"]) for P,lab,cat in POL if f"f_b{B}_p{P}" in runs]
    if not rows: continue
    res = rows[0][2]
    print(f"## 예산 {B} GiB — 상주 {res['resident']:.1f} GiB ({res['res_pct']:.0f}% of experts)\n")
    print("| 정책 | 분류 | **TTFT s** | 프리필 GiB | **TPOT ms** | 디코드 GiB/tok |")
    print("|---|---|---:|---:|---:|---:|")
    bt = min(r["ttft"] for _,_,r in rows); bp = min(r["tpot"] for _,_,r in rows)
    for lab, cat, r in rows:
        t = f"**{r['ttft']:.3f}**" if r["ttft"]==bt else f"{r['ttft']:.3f}"
        p = f"**{r['tpot']:.2f}**" if r["tpot"]==bp else f"{r['tpot']:.2f}"
        print(f"| `{lab}` | {cat} | {t} | {r['pre']:.2f} | {p} | {r['dec']:.4f} |")
    print()

print("## Lookahead — 프리게이팅이 사는 것\n")
print("라우터의 선택을 몇 층 앞서 아는가. 평범한 MoE는 층 L에 닿아야 알고(작은 lookahead), "
      "Pre-gated MoE는 게이트를 한 층 당겨 한 층만큼의 중첩을 산다. "
      "`0`은 한 토큰의 전 층을 한 번에 인출하는 것으로, 드라이버가 원래 모든 정책에 "
      "공짜로 주던 가정이다.\n")
print("| lookahead | `per-layer` TPOT ms | `unified` TPOT ms |")
print("|---:|---:|---:|")
for LA in (1,2,4,8,16):
    a,b = runs.get(f"f_look{LA}_p3"), runs.get(f"f_look{LA}_p8")
    if a or b:
        print(f"| {LA} | {a['tpot']:.2f} | {b['tpot']:.2f} |" if a and b else
              f"| {LA} | {(a or b)['tpot']:.2f} | - |")
for k,lab in (("f_b40_p3","전층(0)"),("f_b40_p8","전층(0)")):
    pass
a,b = runs.get("f_b40_p3"), runs.get("f_b40_p8")
if a and b: print(f"| 전층 (0) | {a['tpot']:.2f} | {b['tpot']:.2f} |")
print()

print("## 교차 도착 — §3.6 전제의 값\n")
print("연속 배치는 프리필과 디코드를 동시에 진행시켜 전역 위상 전환을 불가능하게 한다. "
      "같은 정책을 순차와 교차로 돌린 차이가 그 전제의 값이다.\n")
print("| 정책 | 순차 TTFT | 교차 TTFT | 순차 TPOT | 교차 TPOT | TPOT 배수 |")
print("|---|---:|---:|---:|---:|---:|")
for P,lab,_ in POL:
    a,b = runs.get(f"f_b40_p{P}"), runs.get(f"f_intl_p{P}")
    if a and b:
        print(f"| `{lab}` | {a['ttft']:.3f} | {b['ttft']:.3f} | {a['tpot']:.2f} | "
              f"{b['tpot']:.2f} | {(b['tpot']/a['tpot'] if a['tpot'] else 0):.2f}x |")
