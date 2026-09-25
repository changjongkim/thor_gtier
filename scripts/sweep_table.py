#!/usr/bin/env python3
"""Tabulate results/SIM/sweep and check it against sim_ablate (lru_access,
lfu_all, lfu_decode, phasor_step; miss = 1 - hit) and sim_all's Belady at
the capacities both ran."""
import json, os
S = "results/SIM"; W = ["longbench", "sharegpt", "mmlu"]
out = ["# 용량 스윕 (Qwen3-30B, decode miss rate)", "",
       "LRU는 접근 단위 시각, PHASOR는 스텝 단위 시각(논문 설계), LFU는 prompt를 토큰 수만큼 세는 일반 LFU,",
       "LFU(decode)는 decode 접근만 세는 LFU(PHASOR의 이력 항만 쓴 것, 기준 시스템이 아니라 설계 요소). 정책 규칙은 `sim_ablate.py`와 같다. 이전 판(PHASOR 접근 단위,",
       "LFU decode만)은 `v1_phasor_access_lfu_decode/`에 보관.", ""]
chk = ["## 검증 (sim_ablate / sim_all 대비, miss rate)", "",
       "| 워크로드 | 상주 | 항목 | sweep | 기준 | 차이 |", "|---|---:|---|---:|---:|---:|"]
for w in W:
    pts = json.load(open(f"{S}/sweep/qwen30b_{w}.json"))["points"]
    out += [f"## {w}", "", "| 상주 | LRU | LFU | LFU(decode) | PHASOR | Belady | PHASOR < LRU, LFU? | PHASOR − LFU(decode) |",
            "|---:|---:|---:|---:|---:|---:|---|---:|"]
    for p in pts:
        best = min(p["lru"], p["lfu"])
        tag = "yes" if p["phasor"] <= best else f"no (+{p['phasor'] - best:.3f})"
        out.append(f"| {p['frac']*100:.0f}% | {p['lru']:.3f} | {p['lfu']:.3f} | {p['lfu_decode']:.3f} | "
                   f"**{p['phasor']:.3f}** | {p['belady']:.3f} | {tag} | {p['phasor'] - p['lfu_decode']:+.3f} |")
        for f in ("0.14", "0.35"):
            if abs(p["frac"] - float(f)) > 1e-9: continue
            ab = f"{S}/ablate/qwen30b_{w}_{f}.json"
            if os.path.exists(ab):
                r = json.load(open(ab))["runs"]
                for k, ref in (("lru", "lru_access"), ("lfu", "lfu_all"), ("lfu_decode", "lfu_decode"), ("phasor", "phasor_step")):
                    m = 1 - r[ref]["hit"]
                    chk.append(f"| {w} | {f} | {k} ↔ {ref} | {p[k]:.4f} | {m:.4f} | {p[k]-m:+.4f} |")
            sa = f"{S}/qwen30b_{w}_{f}.json"
            if os.path.exists(sa):
                m = 1 - json.load(open(sa))["Belady"]
                chk.append(f"| {w} | {f} | belady ↔ sim_all | {p['belady']:.4f} | {m:.4f} | {p['belady']-m:+.4f} |")
    out.append("")
open(f"{S}/sweep/SUMMARY.md", "w").write("\n".join(out + chk) + "\n")
print("\n".join(out + chk))
