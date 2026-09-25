# GPU read bandwidth by allocation type (Thor, sm_110)

`tools/memtype.cu`: one kernel streams 384 buffers of 3 MiB (one Qwen3-30B decode step's experts,
1.125 GiB) 20 times; run 2026-09-26 02:26 between two matrix runs (stage 5 paused).

| allocation | GiB/s | ms per step |
|---|---:|---:|
| cudaMalloc | 234.5 | 4.80 |
| cudaHostAlloc(Mapped) — PHASOR arena / staging window | 174.8 | 6.44 |
| cudaHostAlloc(default) | 166.6 | 6.75 |
| cudaMallocManaged | 173.5 | 6.48 |

Reading resident experts from mapped host memory costs 1.6 ms per decode step against device
memory: real (25% lower bandwidth) but two orders of magnitude below the TPOT gap to FlashMoE* at
0.25 (568 vs 355 ms, E1 MMLU), so it is not that gap's cause.
