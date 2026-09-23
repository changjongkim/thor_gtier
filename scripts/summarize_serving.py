#!/usr/bin/env python3
"""Turn the serving queue's per-run files into one document.

Reads whatever finished, so it is useful while the queue is still going.
"""
import glob, os, re, sys

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "SERVE")

RE_HEAD = re.compile(r"budget\s+([\d.]+) GiB\s+window\s+([\d.]+)\s+path-overhead\s+([\d.]+)\s+"
                     r"residency\s+([\d.]+) GiB \(([\d.]+)% of experts\)\s+policy=(\S+)\s+backend=(\S+)")
RE_PIN  = re.compile(r"prefix union: (\d+) units \(([\d.]+) GiB\)")
RE_RES  = re.compile(r"^(\S+)\s+\| TTFT\(io\)\s+([\d.]+) s\s+prefill\s+([\d.]+) GiB \| "
                     r"TPOT\(io\)\s+([\d.]+) ms\s+decode\s+([\d.]+) GiB/tok \| total\s+([\d.]+) GiB")

def parse(path):
    d = {"name": os.path.basename(path)[:-4]}
    for line in open(path):
        m = RE_HEAD.search(line)
        if m:
            d.update(budget=float(m.group(1)), window=float(m.group(2)),
                     overhead=float(m.group(3)), resident=float(m.group(4)),
                     res_pct=float(m.group(5)), policy=m.group(6), backend=m.group(7))
        m = RE_PIN.search(line)
        if m: d.update(pin_units=int(m.group(1)), pin_gib=float(m.group(2)))
        m = RE_RES.match(line)
        if m:
            d.update(ttft=float(m.group(2)), prefill_gib=float(m.group(3)),
                     tpot=float(m.group(4)), dec_gib=float(m.group(5)),
                     total_gib=float(m.group(6)))
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
print("| 프리픽스 토큰 | 핀된 유닛 | 핀 GiB | **TTFT(io) s** | TPOT(io) ms |")
print("|---:|---:|---:|---:|---:|")
for T in (0,10,20,30,50,65):
    k=f"e3_prefix{T}"
    if k in runs:
        r=runs[k]
        print(f"| {T} | {r.get('pin_units','-')} | {r.get('pin_gib',0):.1f} | "
              f"**{r['ttft']:.3f}** | {r['tpot']:.2f} |")
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

print("## E4. 서빙 예산 안에서 스테이징 윈도우 크기\n")
print("윈도우를 키우면 상주가 그만큼 줄어든다. 윈도우는 0.5 GiB 위에서 대역폭을 "
      "더 사주지 않으므로(§4.14) 그 위는 전부 손해여야 한다.\n")
print(H)
for W in ("0.25","0.5","1","2","4"):
    k=f"e4_w{W}"
    if k in runs: print(row(runs[k], f"윈도우 {W} GiB"))
