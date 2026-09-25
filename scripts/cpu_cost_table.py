#!/usr/bin/env python3
"""results/CPU_COST/{micro,real}.log -> results/CPU_COST.md.
Per (backend, item): mean over repeats of GiB/s, process user and sys CPU
seconds per GiB (getrusage, all threads), average cores (process CPU / wall),
and the machine's busy cores above its idle baseline (/proc/stat), which also
catches kernel work outside the process (GPU fault service, interrupts)."""
import re, collections
R = "/home/thor/kcj/thor_gtier/results"
NAME = {"gtier": "gtier(async)", "cufile": "cufile", "pread+copy": "pread+copy",
        "mmap-cpu": "mmap-cpu", "uvm": "uvm", "mmap-gpu": "mmap-gpu"}
ORDER = list(NAME)
NOTES = []


def parse(path):
    rows = collections.defaultdict(list); fails = []; base = 0.0; run = ""
    for ln in open(path):
        if ln.startswith("NOTE"): NOTES.append(ln[5:].strip())
        elif ln.startswith("RUN"): run = ln.strip()
        elif ln.startswith("BASE"): base = float(ln.split("=")[1])
        elif ln.startswith("FAILED"): fails.append(f"{run} {ln.strip()}")
        elif ln.startswith("CPU "):
            kv = dict(x.split("=", 1) for x in ln.split()[1:])
            d = {k: float(v) for k, v in kv.items() if k != "backend"}
            d["net_cores"] = max(0.0, d["machine_cores"] - base)
            d["machine_s_per_gib"] = d["net_cores"] * d["wall_s"] / d["gib"]
            rows[(kv["backend"], int(kv["item"]))].append(d)
    return rows, fails


def size(b):
    return f"{b >> 20} MiB" if b >= 1 << 20 else f"{b >> 10} KiB"


def table(rows, sizes):
    out = ["| backend | read size | GiB/s | user s/GiB | sys s/GiB | avg cores | machine s/GiB | machine cores |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for it in sizes:
        for b in ORDER:
            v = rows.get((b, it))
            if not v: continue
            m = lambda k: sum(x[k] for x in v) / len(v)
            out.append(f"| {NAME[b]} | {size(it)} | {m('gibps'):.3f} | {m('user_s_per_gib'):.3f} | "
                       f"{m('sys_s_per_gib'):.3f} | {m('avg_cores'):.2f} | {m('machine_s_per_gib'):.3f} | "
                       f"{m('net_cores'):.2f} |")
    return out


micro, mf = parse(f"{R}/CPU_COST/micro.log")
real, rf = parse(f"{R}/CPU_COST/real.log")
doc = ["# 데이터 경로별 CPU 비용", "",
       "**측정:** `scripts/cpu_cost.sh` (SSD가 빈 구간, 파이프라인 락 보유) · 원시 로그 `CPU_COST/micro.log`, `CPU_COST/real.log`",
       "",
       "조건은 ASYNC_FIX.md / BACKENDS.md와 같다: 32 GiB 랜덤 파일, 백엔드마다 별도 프로세스에서 페이지 캐시를 비움,",
       "인출당 4 MiB(n = 4 MiB / read size, slot = read size), 16 GiB 구간에 흩어진 오프셋, 실행당 4 GiB, 2회 평균.",
       "",
       "- **user / sys s/GiB**: 시간 측정 구간의 `getrusage(RUSAGE_SELF)` 증가분 ÷ 읽은 GiB. 프로세스의 모든 스레드(io_uring 작업자 포함).",
       "- **avg cores**: (user + sys) ÷ wall.",
       "- **machine s/GiB, machine cores**: `/proc/stat` 전체 busy 시간 증가분에서 실행 직전 3 s 유휴 기준을 뺀 값.",
       "  GPU 페이지 폴트 처리와 완료 인터럽트는 프로세스 밖(커널 스레드, IRQ)에서 돌아 getrusage에 잡히지 않으므로 함께 적는다.",
       "",
       "## 미시 벤치마크 (read size 스윕)", ""] + table(micro, sorted({k[1] for k in micro}))
doc += ["", "## 실제 모델 설정 (Qwen3-30B-A3B bf16, 128 중 8 전문가, skew 0.8, batch 8, window 2 GiB = 512 × 4 MiB, cgroup 8 GiB)", ""]
doc += table(real, sorted({k[1] for k in real}))
if NOTES:
    doc += ["", "## 참고", ""] + [f"- {x}" for x in NOTES]
if mf or rf:
    doc += ["", "## 실패한 실행", ""] + [f"- `{x}`" for x in mf + rf]
open(f"{R}/CPU_COST.md", "w").write("\n".join(doc) + "\n")
print("\n".join(doc))
