#!/usr/bin/env python3
"""Put llama.cpp's own timings next to the bytes it pulled off the device.

llama-bench reports prompt processing and token generation as throughputs;
TTFT and TPOT follow from them.  /proc/<pid>/io gives the read side, and the
ratio of the two is the delivery rate that can be compared against what the
trace driver achieves on the same files.
"""
import glob, json, os, re

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "ENGINE")
GTIER_BW = 4.74      # GiB/s, measured by serve_bench on the same device

def load(path):
    txt = open(path, errors="ignore").read()
    m = re.search(r"IOMETER (\{.*\})", txt)
    io = json.loads(m.group(1)) if m else {}
    rows = []
    # llama-bench emits one JSON array; find it even with progress text around
    for m in re.finditer(r"\[\s*\{.*?\}\s*\]", txt, re.S):
        try:
            rows = json.loads(m.group(0)); break
        except Exception:
            pass
    if not rows:            # fall back: scrape the fields we need
        blocks = re.findall(r'"n_prompt":\s*(\d+),\s*"n_gen":\s*(\d+),.*?"avg_ts":\s*([\d.]+)', txt, re.S)
        rows = [{"n_prompt": int(a), "n_gen": int(b), "avg_ts": float(c)} for a,b,c in blocks]
    return io, rows

print("# 엔진 베이스라인 — llama.cpp on Qwen3-235B-A22B Q4_K_M\n")
print("모델 132.4 GiB 대 메모리 122.8 GiB (M/B = 1.08). 프롬프트 512 토큰, 생성 32 토큰, "
      "실행마다 페이지 캐시 비움.\n")
print("`-ngl`은 GPU에 올릴 층 수인데 이 SoC에서 GPU 메모리는 시스템 메모리와 같으므로, "
      "높은 `-ngl`이 곧 '들어가지 않는 경우'다. `-ncmoe`는 그 층수만큼 전문가를 CPU에 두는 "
      "llama.cpp 자신의 MoE 대응이다.\n")
print("| 설정 | **TTFT s** | **TPOT ms** | 읽은 양 GiB | 벽시계 s | **실효 전달 GiB/s** |")
print("|---|---:|---:|---:|---:|---:|")
for p in sorted(glob.glob(os.path.join(OUT, "e9_*.txt"))):
    name = os.path.basename(p)[:-4]
    io, rows = load(p)
    ttft = tpot = None
    for e in rows:
        ts = e.get("avg_ts", 0)
        if e.get("n_prompt") and ts: ttft = e["n_prompt"]/ts
        if e.get("n_gen") and ts:    tpot = 1000.0/ts
    if ttft is None and tpot is None and not io: continue
    print(f"| {name} | {ttft:.1f} |" if ttft else f"| {name} | - |", end="")
    print(f" {tpot:.0f} |" if tpot else " - |", end="")
    if io:
        print(f" {io['read_GiB']:.1f} | {io['wall_s']:.0f} | **{io['read_GiB_s']:.4f}** |")
    else:
        print(" - | - | - |")
print()
print(f"비교 축: 같은 장치에서 gTier의 전달률은 **{GTIER_BW:.2f} GiB/s**다(§4.16b). "
      "모델이 다르므로 TTFT를 직접 비교할 수는 없고, 전달률이 같은 뜻을 갖는 축이다.\n")
print("읽은 양이 모델 크기를 넘으면 스래싱이다 — 들어가지 않는 페이지를 쫓아냈다가 다시 읽는다.")
