#!/usr/bin/env python3
"""Turn the queue's raw outputs into one table per experiment group."""
import os, re, glob

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "auto")
BACKENDS = {0: "gtier", 1: "mmap-gpu", 2: "mmap-cpu", 3: "pread+copy", 4: "cufile", 5: "uvm"}
POLICIES = {0: "exact", 1: "BLOCK(LRU)", 2: "HYBRID", 3: "ADAPTIVE", 4: "PIN"}

def parse(path):
    try:
        line = open(path).read().strip().splitlines()[-1]
    except Exception:
        return None
    m = re.search(r"^(\S+)\s+policy=(\d).*?window=\s*([\d.]+) GiB \(\s*([\d.]+)% of model\).*?"
                  r"([\d.]+) GiB/s\s+([\d.]+) tok/s(?:\s+hit=([\d.]+)%)?(?:\s+amp=([\d.]+)x)?", line)
    if not m:
        return None
    b, pol, win, winpct, gib, tok, hit, amp = m.groups()
    return dict(backend=b, policy=int(pol), window=float(win), winpct=float(winpct),
                gibs=float(gib), toks=float(tok),
                hit=float(hit) if hit else 0.0, amp=float(amp) if amp else 0.0)

def rows(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(OUT, pattern))):
        d = parse(f)
        if d:
            d["name"] = os.path.basename(f)[:-4]
            out.append(d)
    return out

print("# 자동 실험 큐 요약\n")
print(f"생성: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")
print("유효 대역폭 = 요청한 바이트 ÷ 경과 시간. tok/s = 모델 1회 통과를 1 토큰으로 센 값.\n")

for title, pat, note in [
    ("## A. Dense 모델 (Qwen2.5-32B Q8_0, 32.42 GiB — DRAM에 들어감)", "a_dense_*.txt",
     "토큰 1회 = 모델 전체 1회 통과. 콜드 캐시."),
    ("## B. DRAM 초과 모델 (Qwen3-235B-A22B Q4_K_M, 132.4 GiB vs DRAM 122.8 GiB)", "b_moe_*.txt",
     "이 모델은 DRAM에 들어가지 않는다."),
]:
    r = rows(pat)
    if not r: continue
    print(f"\n{title}\n\n{note}\n")
    print("| 실행 | 백엔드 | GiB/s | tok/s |")
    print("|---|---|---:|---:|")
    for d in sorted(r, key=lambda x: -x["gibs"]):
        print(f"| {d['name']} | {d['backend']} | {d['gibs']:.3f} | {d['toks']:.4f} |")

r = rows("c_moe_res_*.txt")
if r:
    print("\n## C. 상주 정책 — PIN vs LRU (MoE 132.4 GiB, 토큰 3회)\n")
    print("순환 스캔에서 LRU는 윈도우가 워킹셋을 덮기 전까지 이득이 없고, PIN은 비례한다.\n")
    print("| 윈도우 GiB | 모델 대비 | 정책 | GiB/s | tok/s | 히트율 | 증폭 |")
    print("|---:|---:|---|---:|---:|---:|---:|")
    for d in sorted(r, key=lambda x: (x["window"], x["policy"])):
        print(f"| {d['window']:.1f} | {d['winpct']:.1f}% | {POLICIES.get(d['policy'],'?')} | "
              f"{d['gibs']:.3f} | {d['toks']:.4f} | {d['hit']:.1f}% | {d['amp']:.2f}x |")

r = rows("d_moe_gran_*.txt")
if r:
    print("\n## D. 실제 MoE 트레이스 입도별\n")
    print("| 실행 | 백엔드 | GiB/s | tok/s |")
    print("|---|---|---:|---:|")
    for d in sorted(r, key=lambda x: x["name"]):
        print(f"| {d['name']} | {d['backend']} | {d['gibs']:.3f} | {d['toks']:.4f} |")

fails = sorted(glob.glob(os.path.join(OUT, "*.failed")))
if fails:
    print("\n## 실패한 단계\n")
    for f in fails:
        print(f"- `{os.path.basename(f)[:-7]}`")
