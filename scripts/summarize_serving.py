#!/usr/bin/env python3
"""Turn the serving queue's per-run files into one document.

Reads whatever finished, so it is useful while the queue is still going.
"""
import glob, os, re, sys

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "SERVE")

RE_HEAD = re.compile(r"budget\s+(?P<budget>[\d.]+) GiB\s+window\s+(?P<window>[\d.]+)\s+"
                     r"path-overhead\s+(?P<overhead>[\d.]+)\s+residency\s+(?P<resident>[\d.]+) GiB "
                     r"\((?P<res_pct>[\d.]+)% of experts\)\s+policy=(?P<policy>\S+)\s+backend=(?P<backend>\S+)")
RE_PIN  = re.compile(r"prefix union: (?P<pin_units>\d+) units \((?P<pin_gib>[\d.]+) GiB\)")
RE_FAM  = re.compile(r"prefix families:(?P<families>.*)")
RE_EV   = re.compile(r"pins (?P<pins>\d+) evict (?P<evict>\d+)")
RE_STALL= re.compile(r"stall\s+(?P<stall>[\d.]+) s\s+\| bg\s+(?P<bg>[\d.]+) GiB vs shadow\s+(?P<shadow>[\d.]+) GiB")
RE_RES  = re.compile(r"^\S+\s+\| TTFT\(io\)\s+(?P<ttft>[\d.]+) s\s+prefill\s+(?P<prefill_gib>[\d.]+) GiB \| "
                     r"TPOT\(io\)\s+(?P<tpot>[\d.]+) ms\s+decode\s+(?P<dec_gib>[\d.]+) GiB/tok")
RE_TOT  = re.compile(r"total\s+(?P<total_gib>[\d.]+) GiB")
FLOATS  = {"budget","window","overhead","resident","res_pct","pin_gib","ttft",
           "prefill_gib","tpot","dec_gib","stall","bg","shadow","total_gib"}
INTS    = {"pin_units","pins","evict"}

def parse(path):
    d = {"name": os.path.basename(path)[:-4]}
    for line in open(path):
        for rx in (RE_HEAD, RE_PIN, RE_FAM, RE_EV, RE_STALL, RE_RES, RE_TOT):
            m = rx.search(line)
            if not m: continue
            for k, v in m.groupdict().items():
                if v is None: continue
                d[k] = float(v) if k in FLOATS else int(v) if k in INTS else v.strip()
    return d if "ttft" in d else None

runs = {}
for p in sorted(glob.glob(os.path.join(OUT, "*.txt"))):
    if os.path.basename(p) in ("progress.log",): continue
    r = parse(p)
    if r: runs[r["name"]] = r

def row(r, label=None):
    return (f"| {label or r['name']} | {r['policy']} | {r['budget']:.0f} | "
            f"{r['resident']:.1f} ({r['res_pct']:.0f}%) | **{r['ttft']:.3f}** | "
            f"{r['prefill_gib']:.2f} | {r['tpot']:.2f} | {r['dec_gib']:.4f} |")

H = ("| 설정 | 정책 | 예산 GiB | 상주 GiB | **TTFT(io) s** | 프리필 GiB | "
     "TPOT(io) ms | 디코드 GiB/tok |\n|---|---|---:|---:|---:|---:|---:|---:|")

print("# 서빙 실험 요약\n")
print(f"Qwen3-30B-A3B (56.9 GiB, 48층 x 128 전문가, 유닛 6144개 x 9.00 MiB), "
      f"실측 라우팅 트레이스, 실행마다 페이지 캐시 비움.\n")
print(f"완료된 실행 {len(runs)}개.\n")

POL = {"1": "lru", "2": "per-layer", "3": "prefix", "4": "prefix"}

print("## E1. 상주 정책 x 메모리 예산\n")
print("정책은 각각 하나의 결정을 분리한다 — `lru`는 오라클도 위상 개념도 없는 캐시, "
      "`lru+phase`는 거기서 **프리필이 admit하지 않는다는 한 줄만** 다르고, "
      "`per-layer`는 층별 인기순 정적 상주, `prefix`는 거기에 공유 프리픽스 합집합을 핀한다.\n")
print(H)
if "e1_none" in runs: print(row(runs["e1_none"], "상주 없음"))
for B in (12,16,24,32,40,48):
    for P in (1,2,3,4):
        k = f"e1_b{B}_p{P}"
        if k in runs: print(row(runs[k], f"예산 {B}"))
print()

print("## E2. 데이터 경로의 발자국을 같은 예산에 과금하면\n")
print("선언 윈도우 0.5 GiB 위에 각 경로가 실제로 더 쥐는 양을 예산에서 뺀다 "
      "(`results/HF_MOE/footprint.txt` 실측: gtier +0.33, pread+copy +2.35, cuFile +8.62 GiB, "
      "여기서는 2.00 GiB 선언 기준 초과분을 그대로 옮겨 씀).\n")
print(H)
for B in (24,40):
    for nm in ("gtier","pread","cufile"):
        k = f"e2_b{B}_{nm}"
        if k in runs: print(row(runs[k], f"{nm} 예산 {B}"))
print()

print("## E3. 상주를 프리픽스와 인기 사이에 어떻게 나눌 것인가\n")
print("프리픽스로 간주하는 토큰 수를 늘리면 핀되는 합집합이 커지고, "
      "그만큼 층별 인기순 상주가 줄어든다.\n")
print("패밀리는 이름이 아니라 **프리픽스 구간 라우팅 일치율**로 판정한다(>=0.90). "
      "실제로 공유하지 않는 요청은 자동으로 거부되고, 공유 구간을 넘어서면 일치율이 "
      "떨어져 핀을 스스로 포기한다.\n")
print("| 프리픽스 토큰 | 패밀리 일치율 | 핀된 유닛 | 핀 GiB | **TTFT(io) s** | TPOT(io) ms |")
print("|---:|---|---:|---:|---:|---:|")
for T in (0,10,20,30,50,65):
    k=f"e3_prefix{T}"
    if k in runs:
        r=runs[k]
        print(f"| {T} | {r.get('families','-')} | {r.get('pin_units','-')} | "
              f"{r.get('pin_gib',0):.1f} | **{r['ttft']:.3f}** | {r['tpot']:.2f} |")
print()

print("## E5. 오라클 없이 온라인으로 배우면\n")
print("E1의 `per-layer`/`prefix`는 트레이스 전체의 카운트로 순위를 매긴다 — 돌아가는 "
      "시스템에는 없는 정보다. `online`은 층별 LFU를 관측만으로 배우고, `online+prefix`는 "
      "공유 프리픽스를 처음 만난 요청에서 합집합을 배워 핀한다. "
      "오라클과의 간격이 분포를 미리 모르는 비용이다.\n")
print(H)
for B in (16,24,32,40):
    for nm,lab in (("lru","lru"),("online","online"),("onlinep","online+prefix"),("oracle","oracle prefix")):
        k=f"e5_b{B}_{nm}"
        if k in runs: print(row(runs[k], f"{lab} 예산 {B}"))
print()

print("## E6. 시스템 프롬프트가 여럿일 때 — 핀 예산과 축출\n")
print("배포는 시스템 프롬프트를 하나만 쓰지 않는다. 패밀리는 이름이 아니라 "
      "**프리픽스 구간 라우팅 일치율 90% 이상**으로 판정한다. "
      "`--prefix-budget`이 핀 상한이고 0은 무제한이다.\n")
print("| 설정 | 핀 상한 GiB | 핀된 패밀리 | 축출 | **TTFT(io) s** | TPOT(io) ms |")
print("|---|---:|---:|---:|---:|---:|")
for B in (24,40):
    for k,lab in ((f"e6_b{B}_noprefix","프리픽스 없음"),(f"e6_b{B}_single","단일(정적)")):
        if k in runs:
            r=runs[k]
            print(f"| {lab} 예산 {B} | - | {r.get('pins','-')} | {r.get('evict','-')} | "
                  f"**{r['ttft']:.3f}** | {r['tpot']:.2f} |")
    for CAP in (0,4,8,16):
        k=f"e6_b{B}_multi_cap{CAP}"
        if k in runs:
            r=runs[k]
            print(f"| 다중 예산 {B} | {'무제한' if CAP==0 else CAP} | {r.get('pins','-')} | "
                  f"{r.get('evict','-')} | **{r['ttft']:.3f}** | {r['tpot']:.2f} |")
print()

print("## E7. 순차 요청 대 연속 배치\n")
print("위상별 재분할(§3.6)은 시스템이 한 번에 한 위상에 있다고 가정한다. "
      "연속 배치에서는 프리필과 디코드가 동시에 진행되어 전역 전환이 불가능하다. "
      "같은 정책을 순차·교차로 돌린 차이가 그 전제의 값이다.\n")
print("| 정책 | 예산 | 순차 TTFT | 교차 TTFT | 순차 TPOT | 교차 TPOT |")
print("|---|---:|---:|---:|---:|---:|")
for B in (24,40):
    for P,lab in ((1,"lru"),(2,"per-layer"),(4,"prefix"),(7,"multi-prefix")):
        a,b = runs.get(f"e7_b{B}_p{P}_seq"), runs.get(f"e7_b{B}_p{P}_intl")
        if a and b:
            print(f"| {lab} | {B} | {a['ttft']:.3f} | {b['ttft']:.3f} | "
                  f"{a['tpot']:.2f} | {b['tpot']:.2f} |")
print()

print("## E8. 연산이 가릴 수 있는 I/O\n")
print("디코드가 연산 바운드이면 연산 안에 들어가는 I/O는 비용이 아니다. "
      "가정하는 연산 시간을 쓸어 남은 I/O 중 얼마가 실제 스톨인지 본다.\n")
print("| 토큰당 연산 ms | TPOT(io) ms | **스톨 s** | 배경 GiB | 그림자 용량 GiB |")
print("|---:|---:|---:|---:|---:|")
for C in ("0.0","0.89","5","20"):
    k=f"e8_cms{C}"
    if k in runs:
        r=runs[k]
        print(f"| {C} | {r['tpot']:.2f} | **{r.get('stall',0):.2f}** | "
              f"{r.get('bg',0):.2f} | {r.get('shadow',0):.2f} |")
print()

print("## E4. 서빙 예산 안에서 스테이징 윈도우 크기\n")
print("윈도우를 키우면 상주가 그만큼 줄어든다. 윈도우는 0.5 GiB 위에서 대역폭을 "
      "더 사주지 않으므로(§4.14) 그 위는 전부 손해여야 한다.\n")
print(H)
for W in ("0.25","0.5","1","2","4"):
    k=f"e4_w{W}"
    if k in runs: print(row(runs[k], f"윈도우 {W} GiB"))
