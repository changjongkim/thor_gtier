# Architecture-level matrix

Cell: request s / TTFT s / TPOT ms / peak GiB (MemAvailable drop) [request time relative to PHASOR].
Same prompts for every system. Each baseline runs at the knob memcal found to match PHASOR's measured peak at
that budget (two MMLU prompts); every run is capped at 1.05 x that peak + 0.5 GiB. `(over)`: the run's peak
exceeded 1.05 x PHASOR's peak (memory it grew into on longer workloads). `(knob xK)`: the calibrated run hit
the cap and the cell is its retry at K x the calibrated knob. `*` = reimplemented (no released code).
Requests: MMLU 24, ShareGPT 24, LongBench 21 on Qwen3-30B; on Mixtral, ShareGPT and LongBench use their first
12 (the same 12 for every system; a Mixtral run takes 3-10x longer).

## Qwen3-30B-A3B bf16

### MMLU

| system | 25% (14.2 GiB) | 45% (25.7 GiB) | 65% (37.1 GiB) | 108% (61.6 GiB) |
|---|---|---|---|---|
| PHASOR | 9.55 / 7.77 / 254 / 19.3 | 6.68 / 5.46 / 175 / 30.7 | 4.27 / 3.26 / 144 / 42.5 | 2.53 / 1.70 / 118 / 67.4 |
| ZipMoE | 25.16 / 16.47 / 1241 / 20.3 [2.63x] | 18.95 / 12.31 / 948 / 33.3 (over) [2.84x] | 13.10 / 8.12 / 711 / 44.4 [3.06x] | 6.92 / 3.86 / 437 / 70.3 [2.73x] |
| FlashMoE* | 26.26 / 21.47 / 684 / 21.3 (over) [2.75x] | 17.35 / 15.46 / 270 / 33.4 (over) [2.60x] | 11.80 / 10.37 / 204 / 43.3 [2.76x] | 3.19 / 2.42 / 110 / 63.1 [1.26x] |
| DuoServe* | 41.55 / 28.35 / 1887 / 21.2 (over) [4.35x] | 36.03 / 26.38 / 1379 / 32.7 [5.39x] | 32.67 / 25.53 / 1020 / 43.9 [7.64x] | 13.99 / 12.26 / 247 / 65.0 [5.53x] (knob x0.7) |

Reference: each baseline at its own setting for the nominal budget (not equal memory):

| system | 25% | 45% | 65% | 108% |
|---|---|---|---|---|
| ZipMoE (own setting) | 23.47 / 15.44 / 1147 / 24.1 (over) | 18.20 / 11.74 / 924 / 34.6 (over) | 11.62 / 7.15 / 640 / 46.6 (over) | 6.58 / 3.79 / 399 / 70.9 (over) |
| FlashMoE* (own setting) | 24.48 / 20.27 / 601 / 23.4 (over) | 16.47 / 14.45 / 288 / 35.7 (over) | 9.78 / 8.57 / 172 / 48.1 (over) | 3.25 / 2.46 / 113 / 63.3 |
| DuoServe* (own setting) | cannot run (oom-under-cap) | cannot run (oom-under-cap) | cannot run (oom-under-cap) | cannot run (oom-under-cap) |

E7 ablation at 45% (request s relative to PHASOR 6.68 s):

- prefill admitted by value (no free-slot rule): 11.49 s (+72%), TTFT 8.68 s, TPOT 401 ms, peak 30.8 GiB
- LRU (value and admission): 10.23 s (+53%), TTFT 7.15 s, TPOT 439 ms, peak 30.7 GiB
- LRU value, PHASOR admission: 6.72 s (+1%), TTFT 5.41 s, TPOT 187 ms, peak 31.0 GiB
- count utility: 11.23 s (+68%), TTFT 8.20 s, TPOT 432 ms, peak 30.8 GiB
- pread+copy data path (same memory): 7.42 s (+11%), TTFT 6.10 s, TPOT 189 ms, peak 30.9 GiB
- extra device copy (arena kept): 6.79 s (+2%), TTFT 5.64 s, TPOT 165 ms, peak 31.4 GiB
- no intra-layer pipeline: 8.48 s (+27%), TTFT 7.22 s, TPOT 180 ms, peak 30.9 GiB
- no prompt-routing term: 6.46 s (-3%), TTFT 5.26 s, TPOT 172 ms, peak 30.8 GiB
- admit every staged decode unit: 6.44 s (-4%), TTFT 5.24 s, TPOT 171 ms, peak 30.9 GiB

### ShareGPT

| system | 25% (14.2 GiB) | 45% (25.7 GiB) | 65% (37.1 GiB) | 108% (61.6 GiB) |
|---|---|---|---|---|
| PHASOR | 21.08 / 7.57 / 436 / 19.3 | 10.74 / 4.94 / 187 / 30.9 | 7.27 / 3.00 / 138 / 42.5 | 5.25 / 1.64 / 117 / 67.5 |
| ZipMoE | 57.67 / 15.46 / 1362 / 23.3 (over) [2.74x] | 47.01 / 12.24 / 1122 / 35.0 (over) [4.38x] | 33.65 / 8.10 / 824 / 47.0 (over) [4.63x] | 16.96 / 3.78 / 425 / 72.2 (over) [3.23x] |
| FlashMoE* | 48.26 / 19.76 / 919 / 22.3 (over) [2.29x] | 27.67 / 13.88 / 445 / 34.0 (over) [2.58x] | 19.24 / 10.04 / 297 / 43.9 [2.65x] | 5.92 / 2.38 / 114 / 63.9 [1.13x] |
| DuoServe* | 79.10 / 27.62 / 1661 / 20.9 (over) [3.75x] | 54.51 / 25.59 / 933 / 32.9 [5.08x] | 38.43 / 20.74 / 571 / 42.2 [5.29x] | 17.54 / 9.77 / 251 / 65.3 [3.34x] (knob x0.7) |

E7 ablation at 45% (request s relative to PHASOR 10.74 s):

- prefill admitted by value (no free-slot rule): 16.60 s (+55%), TTFT 8.97 s, TPOT 246 ms, peak 30.9 GiB
- LRU (value and admission): 17.02 s (+58%), TTFT 8.61 s, TPOT 271 ms, peak 31.5 GiB
- LRU value, PHASOR admission: 11.09 s (+3%), TTFT 4.96 s, TPOT 198 ms, peak 30.9 GiB
- count utility: 17.10 s (+59%), TTFT 8.75 s, TPOT 269 ms, peak 30.9 GiB
- pread+copy data path (same memory): 12.27 s (+14%), TTFT 5.58 s, TPOT 216 ms, peak 31.0 GiB
- extra device copy (arena kept): 10.63 s (-1%), TTFT 5.18 s, TPOT 176 ms, peak 31.4 GiB
- no intra-layer pipeline: 12.46 s (+16%), TTFT 6.57 s, TPOT 190 ms, peak 30.9 GiB
- no prompt-routing term: 10.81 s (+1%), TTFT 4.89 s, TPOT 191 ms, peak 31.5 GiB
- admit every staged decode unit: 10.68 s (-1%), TTFT 4.81 s, TPOT 189 ms, peak 30.9 GiB

### LongBench

| system | 25% (14.2 GiB) | 45% (25.7 GiB) | 65% (37.1 GiB) | 108% (61.6 GiB) |
|---|---|---|---|---|
| PHASOR | 23.75 / 11.00 / 411 / 20.3 | 16.47 / 9.14 / 237 / 31.9 | 11.91 / 6.79 / 165 / 43.5 | 7.88 / 4.36 / 114 / 68.5 |
| ZipMoE | 80.61 / 30.18 / 1627 / 58.4 (over) [3.39x] | 64.71 / 26.31 / 1239 / 68.6 (over) [3.93x] | 48.63 / 21.78 / 866 / 81.7 (over) [4.08x] | 29.80 / 17.16 / 408 / 97.7 (over) [3.78x] (knob x0.85) |
| FlashMoE* | 55.52 / 23.91 / 1020 / 24.1 (over) [2.34x] | 38.49 / 19.02 / 628 / 34.3 (over) [2.34x] | 24.45 / 13.94 / 339 / 45.3 (over) [2.05x] | 8.24 / 4.74 / 113 / 64.5 [1.05x] |
| DuoServe* | 85.25 / 32.51 / 1701 / 22.4 (over) [3.59x] | 61.52 / 31.52 / 968 / 32.8 [3.74x] | 46.10 / 26.87 / 620 / 45.0 (over) [3.87x] | 29.55 / 20.46 / 293 / 65.8 [3.75x] (knob x0.7) |

E7 ablation at 45% (request s relative to PHASOR 16.47 s):

- prefill admitted by value (no free-slot rule): 20.54 s (+25%), TTFT 12.86 s, TPOT 248 ms, peak 31.9 GiB
- LRU (value and admission): 20.70 s (+26%), TTFT 12.35 s, TPOT 270 ms, peak 31.9 GiB
- LRU value, PHASOR admission: 16.60 s (+1%), TTFT 8.94 s, TPOT 247 ms, peak 31.9 GiB
- count utility: 21.68 s (+32%), TTFT 12.49 s, TPOT 297 ms, peak 31.9 GiB
- pread+copy data path (same memory): 16.96 s (+3%), TTFT 8.54 s, TPOT 272 ms, peak 32.0 GiB
- extra device copy (arena kept): 15.90 s (-3%), TTFT 9.14 s, TPOT 218 ms, peak 32.4 GiB
- no intra-layer pipeline: 17.77 s (+8%), TTFT 10.43 s, TPOT 237 ms, peak 31.9 GiB
- no prompt-routing term: 16.63 s (+1%), TTFT 8.92 s, TPOT 249 ms, peak 31.9 GiB
- admit every staged decode unit: 16.28 s (-1%), TTFT 9.05 s, TPOT 233 ms, peak 31.9 GiB

### E2: small budgets (MMLU, each system at its own setting)

| system | 20% (11.40 GiB) | 15% (8.55 GiB) | 10% (5.70 GiB) | 5% (2.85 GiB) |
|---|---|---|---|---|
| PHASOR | 11.28 / 8.72 / 367 / 16.3 | 13.53 / 9.25 / 611 / 13.4 | 15.28 / 9.91 / 767 / 11.1 | 17.03 / 10.37 / 952 / 7.7 |
| ZipMoE | 24.37 / 16.09 / 1182 / 20.4 | 26.83 / 17.68 / 1306 / 16.8 | 27.11 / 17.94 / 1309 / 15.3 | 29.15 / 19.49 / 1380 / 12.0 |
| FlashMoE* | 27.47 / 22.13 / 763 / 20.2 | 30.28 / 23.55 / 962 / 16.9 | 33.26 / 24.55 / 1244 / 13.9 | 39.60 / 26.92 / 1811 / 9.2 |
| DuoServe* | 41.95 / 28.76 / 1884 / 21.1 | 44.47 / 29.13 / 2191 / 18.5 | 48.99 / 29.16 / 2834 / 13.4 | 58.34 / 29.64 / 4100 / 9.9 |

### E5: PHASOR latency breakdown at 45%

| workload | phase | steps | I/O wait | expert matmuls | other MoE | unit hit rate |
|---|---|---:|---:|---:|---:|---:|
| MMLU | prefill (s/request) | 24 | 0.50 | 2.41 | 2.40 | 0.559 |
| MMLU | decode (ms/step) | 168 | 30.34 | 75.39 | 22.09 | 0.965 |
| ShareGPT | prefill (s/request) | 24 | 0.47 | 2.17 | 2.20 | 0.578 |
| ShareGPT | decode (ms/step) | 744 | 41.29 | 72.88 | 25.88 | 0.953 |
| LongBench | prefill (s/request) | 21 | 0.39 | 3.48 | 3.82 | 0.493 |
| LongBench | decode (ms/step) | 651 | 67.09 | 78.67 | 38.56 | 0.922 |

### E9: staging window at 45% (PHASOR, request s / TTFT s / TPOT ms / peak GiB)

- MMLU: E1 window: 6.68 / 5.46 / 175 / 30.7; window 0.25 GiB: 6.69 / 5.52 / 167 / 30.9; window 1 GiB: 6.56 / 5.33 / 176 / 30.8; window 2 GiB: 7.00 / 5.72 / 182 / 30.8
- ShareGPT: E1 window: 10.74 / 4.94 / 187 / 30.9; window 0.25 GiB: 10.61 / 5.04 / 180 / 30.8; window 1 GiB: 10.85 / 4.81 / 195 / 30.9; window 2 GiB: 11.54 / 5.23 / 203 / 30.9

### E3: batching at 45% (tokens/s; group request s)

| workload | system | batch 1 | batch 4 | batch 8 |
|---|---|---:|---:|---:|
| MMLU | PHASOR | 1.20; 6.68 | 3.13; 10.22 | 4.24; 15.09 |
| MMLU | ZipMoE | 0.42; 18.95 | 1.39; 23.05 | 1.97; 32.57 |
| MMLU | FlashMoE* | 0.46; 17.35 | 1.39; 22.99 | 1.87; 34.14 |
| MMLU | DuoServe* | 0.22; 36.03 | 0.73; 44.07 | 1.15; 55.65 |
| ShareGPT | PHASOR | 2.98; 10.74 | 4.59; 27.90 | 4.79; 53.43 |
| ShareGPT | ZipMoE | 0.68; 47.01 | 1.37; 93.77 | 1.76; 145.45 |
| ShareGPT | FlashMoE* | 1.16; 27.67 | 1.94; 65.93 | 2.46; 104.09 |
| ShareGPT | DuoServe* | 0.59; 54.51 | 1.18; 108.77 | 1.40; 182.76 |

## Mixtral-8x7B bf16

### MMLU

| system | 25% (21.8 GiB) | 45% (39.1 GiB) | 65% (56.6 GiB) | 108% (94.0 GiB) |
|---|---|---|---|---|
| PHASOR | 40.53 / 14.71 / 3688 / 26.8 | 27.91 / 11.51 / 2344 / 44.4 | 17.08 / 8.12 / 1280 / 54.5 | cannot run (host-safety-ceiling) |
| ZipMoE | 58.53 / 20.86 / 5382 / 26.7 [1.44x] | 40.64 / 14.20 / 3778 / 45.6 [1.46x] | 25.69 / 10.77 / 2132 / 64.1 [1.50x] | cannot run (host-safety-ceiling) |
| FlashMoE* | 97.51 / 36.94 / 8653 / 20.7 [2.41x] | 60.41 / 26.84 / 4795 / 41.7 [2.16x] | 32.05 / 17.87 / 2024 / 61.0 [1.88x] | cannot run (host-safety-ceiling) |
| DuoServe* | 188.93 / 50.90 / 19719 / 25.0 [4.66x] | 138.84 / 42.18 / 13809 / 43.2 [4.97x] | 99.48 / 38.39 / 8727 / 62.2 [5.82x] | cannot run (host-safety-ceiling) |
| Fiddler | cannot run (oom-under-cap) | - | - | - |
| Mixtral-offloading (2-bit) | 7.65 / 5.10 / 364 / 22.7 [0.19x] | - | - | - |

Reference: each baseline at its own setting for the nominal budget (not equal memory):

| system | 25% | 45% | 65% | 108% |
|---|---|---|---|---|
| ZipMoE (own setting) | 50.77 / 16.31 / 4923 / 35.6 (over) | - | - | - |
| FlashMoE* (own setting) | 78.52 / 32.08 / 6635 / 29.7 (over) | - | - | - |

### ShareGPT

| system | 25% (21.8 GiB) | 45% (39.1 GiB) | 65% (56.6 GiB) | 108% (94.0 GiB) |
|---|---|---|---|---|

### LongBench

| system | 25% (21.8 GiB) | 45% (39.1 GiB) | 65% (56.6 GiB) | 108% (94.0 GiB) |
|---|---|---|---|---|

### E2: small budgets (MMLU, each system at its own setting)

| system | 20% (17.40 GiB) | 15% (13.05 GiB) | 10% (8.70 GiB) | 5% (4.35 GiB) |
|---|---|---|---|---|

### E5: PHASOR latency breakdown at 45%

| workload | phase | steps | I/O wait | expert matmuls | other MoE | unit hit rate |
|---|---|---:|---:|---:|---:|---:|

### E9: staging window at 45% (PHASOR, request s / TTFT s / TPOT ms / peak GiB)


### E3: batching at 45% (tokens/s; group request s)

| workload | system | batch 1 | batch 4 | batch 8 |
|---|---|---:|---:|---:|

