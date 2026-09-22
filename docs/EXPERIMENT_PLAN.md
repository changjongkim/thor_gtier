# Out-of-core LLM inference on coherent edge SoCs

**Platform:** Jetson AGX Thor, JetPack 7.2 / L4T R39.2, CUDA 13.0, sm_110,
122.8 GiB unified coherent memory, WD SN5000S 1 TB NVMe (4.9 GB/s read), no swap.

## 1. The claim

Every system for running a model larger than memory assumes **three distinct
tiers** — GPU device memory, host DRAM, and storage — and spends its design
budget scheduling movement between tiers 1 and 2. On a coherent edge SoC tiers 1
and 2 are *the same physical memory*: `llama.cpp` on this device reports
`Total VRAM: 125748 MiB`, which is all of system RAM. The GPU/host split that
those systems optimize has no physical meaning here, and the entire design space
collapses onto a single DRAM↔flash boundary that none of them targets.

The hardware's own answer to that boundary is new: Thor is the first Tegra with
`pageableMemoryAccessUsesHostPageTables=1`, so a GPU kernel can dereference
file-backed `mmap` memory and the OS services the faults. We measured that path
(`../mmap_gpu/`) and it is not usable:

| Working set | Result | Effective BW | % of NVMe |
|---:|---|---:|---:|
| 8–16 GiB | OK | 1.29–1.35 GiB/s | 26–28% |
| 20–48 GiB | OK | 0.21–0.36 GiB/s | 4–7% |
| 64 GiB | **crash** (`illegal memory access`) | — | — |
| 80 GiB | OK | 0.19 GiB/s | 4% |
| 100, 200 GiB | **crash** | — | — |

Three regimes: a 3.6× performance cliff at 16→20 GiB, a ceiling at 4–28% of
device bandwidth, and **nondeterministic hard failure** (64 GiB fails, 80 GiB
succeeds) — a race in reclaim/invalidation, not a capacity limit. With no swap,
there is no fallback. The failure mode is the worst kind: it works, then dies.

## 2. Baselines — all top-tier, and what each assumes

| System | Venue | Premise | Status on a coherent SoC |
|---|---|---|---|
| ZeRO-Infinity | SC'21 | GPU HBM → CPU DRAM → NVMe partitioning | tiers 1–2 identical; the partition is a no-op |
| FlashNeuron | **FAST'21** | GPU memory ↔ SSD via GPUDirect, bypassing CPU | no separate GPU memory; cuFile on Jetson appears to run in POSIX compat mode |
| DeepUM | ASPLOS'23 | UVM page migration + correlation prefetch | there is nowhere to migrate *to* |
| FlexGen | ICML'23 | search over a 3-level GPU/CPU/disk block schedule | two of three levels merge; the search space degenerates |
| G10 | MICRO'23 | unified GPU+host+flash space, compiler-guided tensor migration | closest in spirit; assumes discrete GPU memory as tier 1, and is evaluated in simulation |
| PowerInfer | SOSP'24 | hot neurons resident on GPU, cold on CPU | same memory → the split changes only where compute runs |
| InfiniGen | OSDI'24 | speculative KV offload GPU→CPU | data movement is a no-op |
| NEO | MLSys'25 | offload attention/KV to CPU | same |
| InstInfer / INF2 | 2024–25 | in-/near-storage attention offload | orthogonal; requires computational storage |
| LLM in a Flash | ACL'24 | flash→DRAM windowing, row-column bundling, sparsity-aware loading | **nearest work.** Unified memory, flash-aware. But it decides *what* to load on the CPU side; it does not use or measure GPU-visible demand paging, and does not address the reclaim failure |

The boundary to defend: prior work optimizes **what** to move between a GPU and a
host that are physically distinct. Here they are not, and the open problem is the
**mechanism** of the one remaining boundary — GPU-visible paging against flash —
which the hardware now exposes and which does not work.

## 3. Workload design

Quantization is used purely as a **size knob**, so the model architecture,
routing and access pattern stay fixed while the oversubscription ratio sweeps
across the DRAM boundary.

**Primary — Qwen3-235B-A22B-Instruct (MoE, 22B active per token).**
Only ~9% of weights are read per token, scattered by expert routing. This is the
worst case for the kernel's sequential readahead and the best case for a
structure-aware tier.

| Quant | Size | vs 122.8 GiB |
|---|---:|---:|
| Q3_K_M | 104.7 GiB | 0.86× (fits) |
| Q4_K_M | 132.4 GiB | 1.09× (exceeds) |
| Q5_K_M | 155.4 GiB | 1.27× |
| Q6_K | 179.8 GiB | 1.47× |
| Q8_0 | 232.8 GiB | 1.91× |

**Secondary — Qwen2.5 dense ladder, Q8_0:** 7B (8 GiB), 14B (16 GiB), 32B
(35 GiB), 72B (77 GiB). These land on the measured mmap regimes (fast / cliff /
slow / slow) and give a dense-vs-MoE access-pattern contrast.

## 4. Experiments

**E1 — the `-ngl` sweep (falsifies the three-tier premise).**
On a discrete GPU, `-ngl K` decides how many layers sit in VRAM versus streaming
from host memory per token, and dominates both throughput and PCIe traffic. On
Thor, if bytes-read and page-fault counts are flat across `-ngl` while only
compute placement changes, the GPU/host tier distinction is empirically dead on
this hardware.

**E2 — crossing the DRAM boundary.** Sweep the MoE quant ladder through 1.0×.
Report throughput, TTFT, major faults, NVMe read volume, page-cache growth, and
whether it completes at all.

**E3 — mechanism comparison.** `mmap` (OS demand paging) vs `--no-mmap`
(explicit read) vs a bounded-window prototype. `mmap` is the path characterized
in §1; `--no-mmap` cannot exceed DRAM at all.

**E4 — MoE vs dense.** Same bytes, different access structure. Does the kernel's
readahead help or hurt when the access is expert-scattered?

**Metrics** (all captured by `bench.py`): throughput (pp/tg), wall time,
`pgmajfault`, `pgfault`, NVMe sectors read/written, page-cache growth,
MemAvailable, board power.

## 5. Status

- `bench.py` written and validated end-to-end on TinyLlama-1.1B.
- llama.cpp built for sm_110, confirmed working on Thor (TinyLlama: pp64 4175 t/s,
  tg32 226 t/s at `-ngl 99`).
- Downloads running: Qwen2.5 7B/14B/32B/72B Q8_0, then Qwen3-235B Q3_K_M and
  Q4_K_M. ~23.8 MB/s from HF; several hours total.

## 6. Go/kill

The paper exists only if a bounded-window userspace tier both (a) removes the
nondeterministic failure and (b) beats 1.35 GiB/s — ideally approaching the
4.9 GB/s the device can actually deliver. If the ceiling turns out to be
hardware, this reduces to a characterization paper.
