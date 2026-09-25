# Serving matrix — each system on its own data path

Same prompts on every model. Every run under a cgroup cap of budget + 0.5 GiB (pinned host memory and page cache counted). I/O measured; per-token compute calibrated with llama-bench (results/MATRIX/calib.tsv) except llama.cpp, which is measured end to end. Budgets are fractions of the model's bytes.

## qwen30b

### qwen30b / longbench

| budget | system | request s | TTFT s | TPOT ms | tok/s | peak GiB |
|---|---|---|---|---|---|---|
| 0.25 | LRU (pread+copy) | 13.511 | 8.408 | 159.5 | 2.37 | 3.37 |
| 0.25 | MoE-Infinity* (pread+copy, prefetch) | 10.616 | 5.471 | 160.8 | 3.01 | 3.37 |
| 0.25 | Mixtral-offloading* (copy, speculative) | 10.172 | 5.482 | 146.6 | 3.15 | 3.37 |
| 0.25 | PHASOR | 8.025 | 5.331 | 84.2 | 3.99 | 3.87 |
| 0.25 | **PHASOR vs best other (mixtral)** | **1.27x** | 1.03x | 1.74x | | |
| 0.45 | LRU (pread+copy) | 10.054 | 7.502 | 79.8 | 3.18 | 6.85 |
| 0.45 | MoE-Infinity* (pread+copy, prefetch) | 7.160 | 5.303 | 58.0 | 4.47 | 6.85 |
| 0.45 | Mixtral-offloading* (copy, speculative) | 7.208 | 5.303 | 59.5 | 4.44 | 6.85 |
| 0.45 | llama.cpp (--cpu-moe, mmap) | 38.898 | 32.903 | 193.4 | 0.82 | 8.28 |
| 0.45 | PHASOR | 6.150 | 5.215 | 29.2 | 5.20 | 7.35 |
| 0.45 | **PHASOR vs best other (moeinf)** | **1.16x** | 1.02x | 1.99x | | |
| 0.65 | LRU (pread+copy) | 8.180 | 6.602 | 49.3 | 3.91 | 10.32 |
| 0.65 | MoE-Infinity* (pread+copy, prefetch) | 6.304 | 5.218 | 33.9 | 5.08 | 10.32 |
| 0.65 | Mixtral-offloading* (copy, speculative) | 6.407 | 5.208 | 37.4 | 4.99 | 10.32 |
| 0.65 | PHASOR | 5.704 | 5.190 | 16.1 | 5.61 | 10.82 |
| 0.65 | **PHASOR vs best other (moeinf)** | **1.11x** | 1.01x | 2.11x | | |

Batch 4 at 0.45:

| system | request s | TTFT s | TPOT ms | tok/s |
|---|---|---|---|---|
| MoE-Infinity* batch 4 | 30.180 | 20.341 | 307.5 | 4.16 |
| PHASOR batch 4 | 25.515 | 20.325 | 162.2 | 4.92 |

Ablation at 0.45:

| configuration | request s | TTFT s | TPOT ms |
|---|---|---|---|
| PHASOR | 6.150 | 5.215 | 29.2 |
| - layer pipelining | 8.047 | 6.733 | 41.1 |
| - continuous submission | 6.442 | 5.283 | 36.2 |
| pread+copy path instead of gTier | 6.674 | 5.314 | 42.5 |
| LRU residency on gTier path | 6.606 | 5.217 | 43.4 |
| - prompt routing term | 6.321 | 5.218 | 34.5 |
| - decode history term | 6.063 | 5.216 | 26.5 |
| - recency term | 6.877 | 5.225 | 51.6 |
| count utility | 6.913 | 5.227 | 52.7 |

### qwen30b / sharegpt

| budget | system | request s | TTFT s | TPOT ms | tok/s | peak GiB |
|---|---|---|---|---|---|---|
| 0.25 | LRU (pread+copy) | 8.003 | 2.987 | 156.8 | 4.00 | 3.20 |
| 0.25 | MoE-Infinity* (pread+copy, prefetch) | 7.290 | 2.460 | 151.0 | 4.39 | 3.20 |
| 0.25 | Mixtral-offloading* (copy, speculative) | 6.963 | 2.448 | 141.1 | 4.60 | 3.20 |
| 0.25 | PHASOR | 4.577 | 1.728 | 89.0 | 6.99 | 3.70 |
| 0.25 | **PHASOR vs best other (mixtral)** | **1.52x** | 1.42x | 1.58x | | |
| 0.45 | LRU (pread+copy) | 4.933 | 2.197 | 85.5 | 6.49 | 6.68 |
| 0.45 | MoE-Infinity* (pread+copy, prefetch) | 3.751 | 1.699 | 64.1 | 8.53 | 6.68 |
| 0.45 | Mixtral-offloading* (copy, speculative) | 3.867 | 1.691 | 68.0 | 8.28 | 6.68 |
| 0.45 | llama.cpp (--cpu-moe, mmap) | 21.642 | 12.440 | 296.8 | 1.48 | 8.28 |
| 0.45 | PHASOR | 2.019 | 1.147 | 27.2 | 15.85 | 7.18 |
| 0.45 | **PHASOR vs best other (moeinf)** | **1.86x** | 1.48x | 2.36x | | |
| 0.65 | LRU (pread+copy) | 2.973 | 1.500 | 46.0 | 10.76 | 10.15 |
| 0.65 | MoE-Infinity* (pread+copy, prefetch) | 1.810 | 1.041 | 24.0 | 17.68 | 10.15 |
| 0.65 | Mixtral-offloading* (copy, speculative) | 2.123 | 1.005 | 35.0 | 15.07 | 10.15 |
| 0.65 | PHASOR | 1.160 | 0.682 | 15.0 | 27.59 | 10.66 |
| 0.65 | **PHASOR vs best other (moeinf)** | **1.56x** | 1.53x | 1.61x | | |

Batch 4 at 0.45:

| system | request s | TTFT s | TPOT ms | tok/s |
|---|---|---|---|---|
| MoE-Infinity* batch 4 | 9.085 | 2.649 | 201.1 | 14.09 |
| PHASOR batch 4 | 5.124 | 2.309 | 88.0 | 24.98 |

Ablation at 0.45:

| configuration | request s | TTFT s | TPOT ms |
|---|---|---|---|
| PHASOR | 2.019 | 1.147 | 27.2 |
| - layer pipelining | 2.890 | 1.665 | 38.3 |
| - continuous submission | 2.575 | 1.521 | 32.9 |
| pread+copy path instead of gTier | 2.993 | 1.689 | 40.8 |
| LRU residency on gTier path | 2.877 | 1.206 | 52.2 |
| - prompt routing term | 2.033 | 1.179 | 26.7 |
| - decode history term | 2.206 | 1.167 | 32.5 |
| - recency term | 2.324 | 1.169 | 36.1 |
| count utility | 2.194 | 1.182 | 31.6 |

### qwen30b / mmlu

| budget | system | request s | TTFT s | TPOT ms | tok/s | peak GiB |
|---|---|---|---|---|---|---|
| 0.25 | LRU (pread+copy) | 4.440 | 3.109 | 166.4 | 1.80 | 3.19 |
| 0.25 | MoE-Infinity* (pread+copy, prefetch) | 3.912 | 2.661 | 156.4 | 2.05 | 3.19 |
| 0.25 | Mixtral-offloading* (copy, speculative) | 3.865 | 2.653 | 151.5 | 2.07 | 3.20 |
| 0.25 | PHASOR | 2.467 | 1.911 | 69.5 | 3.24 | 3.71 |
| 0.25 | **PHASOR vs best other (mixtral)** | **1.57x** | 1.39x | 2.18x | | |
| 0.45 | LRU (pread+copy) | 3.588 | 2.310 | 159.7 | 2.23 | 6.68 |
| 0.45 | MoE-Infinity* (pread+copy, prefetch) | 2.131 | 1.903 | 28.6 | 3.75 | 6.68 |
| 0.45 | Mixtral-offloading* (copy, speculative) | 2.838 | 1.878 | 120.0 | 2.82 | 6.68 |
| 0.45 | llama.cpp (--cpu-moe, mmap) | 23.866 | 19.370 | 642.3 | 0.34 | 8.28 |
| 0.45 | PHASOR | 1.469 | 1.306 | 20.4 | 5.44 | 7.18 |
| 0.45 | **PHASOR vs best other (moeinf)** | **1.45x** | 1.46x | 1.40x | | |
| 0.65 | LRU (pread+copy) | 1.804 | 1.557 | 30.9 | 4.43 | 10.14 |
| 0.65 | MoE-Infinity* (pread+copy, prefetch) | 1.341 | 1.212 | 16.1 | 5.96 | 10.06 |
| 0.65 | Mixtral-offloading* (copy, speculative) | 1.300 | 1.126 | 21.8 | 6.16 | 10.06 |
| 0.65 | PHASOR | 0.831 | 0.723 | 13.6 | 9.63 | 10.55 |
| 0.65 | **PHASOR vs best other (mixtral)** | **1.56x** | 1.56x | 1.61x | | |

Batch 4 at 0.45:

| system | request s | TTFT s | TPOT ms | tok/s |
|---|---|---|---|---|
| MoE-Infinity* batch 4 | 3.074 | 2.413 | 82.6 | 10.41 |
| PHASOR batch 4 | 2.284 | 2.016 | 33.4 | 14.01 |

Ablation at 0.45:

| configuration | request s | TTFT s | TPOT ms |
|---|---|---|---|
| PHASOR | 1.469 | 1.306 | 20.4 |
| - layer pipelining | 1.959 | 1.724 | 29.4 |
| - continuous submission | 1.884 | 1.696 | 23.4 |
| pread+copy path instead of gTier | 2.104 | 1.882 | 27.8 |
| LRU residency on gTier path | 2.158 | 1.304 | 106.8 |
| - prompt routing term | 1.417 | 1.276 | 17.7 |
| - decode history term | 1.627 | 1.315 | 39.0 |
| - recency term | 1.464 | 1.299 | 20.7 |
| count utility | 1.493 | 1.338 | 19.4 |

