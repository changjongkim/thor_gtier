# Serving matrix

Same prompts on every model; request time = prefill I/O + prompt compute + decode (I/O + compute) per token, averaged over requests.  I/O measured, compute calibrated (results/MATRIX/calib.tsv).  Budgets are fractions of the model's bytes.

## Compute calibration

```
model	decode_ms	prompt_ms	source
qwen30b	13.4466	0.81486	llama-bench ngl99
mixtral8x7b	38.6528	1.68858	llama-bench ngl99
qwen235b	86.2574	5.22716	qwen30b x 6.415 (bytes/token 1.344e+10/2.095e+09)
```

## qwen30b

### qwen30b / longbench

| budget | policy | request s | TTFT s | TPOT ms | tok/s (e2e) | prefill GiB/req | decode GiB/tok |
|---|---|---|---|---|---|---|---|
| 0.25 (4.3 GiB) | LRU | 10.556 | 7.189 | 105.23 | 3.031 | 11.11 | 0.2911 |
| 0.25 (4.3 GiB) | MoE-Infinity* | 10.899 | 7.203 | 115.48 | 2.936 | 11.10 | 0.3243 |
| 0.25 (4.3 GiB) | Mixtral-offloading* | 10.539 | 7.180 | 104.97 | 3.036 | 11.11 | 0.2929 |
| 0.25 (4.3 GiB) | LEDGER | 10.367 | 7.209 | 98.68 | 3.087 | 11.21 | 0.2730 |
| 0.25 | **LEDGER vs best baseline (mixtral*)** | **1.02x** | 1.00x | 1.06x | | | |
| 0.45 (7.8 GiB) | LRU | 8.564 | 6.630 | 60.45 | 3.736 | 7.99 | 0.1461 |
| 0.45 (7.8 GiB) | MoE-Infinity* | 8.457 | 6.626 | 57.24 | 3.784 | 7.98 | 0.1315 |
| 0.45 (7.8 GiB) | Mixtral-offloading* | 8.556 | 6.632 | 60.12 | 3.740 | 8.00 | 0.1444 |
| 0.45 (7.8 GiB) | LEDGER | 8.046 | 6.630 | 44.22 | 3.977 | 7.98 | 0.0865 |
| 0.45 | **LEDGER vs best baseline (moe-inf*)** | **1.05x** | 1.00x | 1.29x | | | |
| 0.65 (11.2 GiB) | LRU | 7.387 | 6.123 | 39.50 | 4.332 | 5.15 | 0.0732 |
| 0.65 (11.2 GiB) | MoE-Infinity* | 7.350 | 6.121 | 38.42 | 4.354 | 5.09 | 0.0698 |
| 0.65 (11.2 GiB) | Mixtral-offloading* | 7.390 | 6.124 | 39.56 | 4.330 | 5.15 | 0.0732 |
| 0.65 (11.2 GiB) | LEDGER | 6.840 | 6.066 | 24.20 | 4.678 | 4.81 | 0.0249 |
| 0.65 | **LEDGER vs best baseline (moe-inf*)** | **1.07x** | 1.01x | 1.59x | | | |

Ablation at 0.45 (qwen30b / longbench):

| configuration | request s | TTFT s | TPOT ms |
|---|---|---|---|
| LEDGER (full) | 8.046 | 6.630 | 44.22 |
| - async submission | 8.749 | 7.089 | 51.88 |
| - prompt routing term (history only) | 8.239 | 6.625 | 50.47 |
| - decode history term (prompt only) | 7.908 | 6.624 | 40.12 |
| - recency term | 8.879 | 6.642 | 69.91 |
| - selective admission | 8.004 | 6.621 | 43.23 |
| count utility instead of estimate | 8.892 | 6.636 | 70.52 |
| + held-out initial counts | 7.931 | 6.592 | 41.82 |
| - prefix pin | 7.995 | 6.618 | 43.05 |
| - live-set guard | 7.981 | 6.615 | 42.68 |
| no residency | 15.661 | 7.636 | 250.80 |

### qwen30b / sharegpt

| budget | policy | request s | TTFT s | TPOT ms | tok/s (e2e) | prefill GiB/req | decode GiB/tok |
|---|---|---|---|---|---|---|---|
| 0.25 (4.3 GiB) | LRU | 5.973 | 2.200 | 117.92 | 5.357 | 9.18 | 0.3536 |
| 0.25 (4.3 GiB) | MoE-Infinity* | 6.160 | 2.203 | 123.64 | 5.195 | 9.19 | 0.3752 |
| 0.25 (4.3 GiB) | Mixtral-offloading* | 5.934 | 2.206 | 116.50 | 5.393 | 9.18 | 0.3486 |
| 0.25 (4.3 GiB) | LEDGER | 5.741 | 2.208 | 110.41 | 5.574 | 9.23 | 0.3215 |
| 0.25 | **LEDGER vs best baseline (mixtral*)** | **1.03x** | 1.00x | 1.06x | | | |
| 0.45 (7.8 GiB) | LRU | 3.942 | 1.656 | 71.43 | 8.118 | 6.01 | 0.1831 |
| 0.45 (7.8 GiB) | MoE-Infinity* | 3.456 | 1.648 | 56.49 | 9.260 | 6.05 | 0.1294 |
| 0.45 (7.8 GiB) | Mixtral-offloading* | 3.820 | 1.653 | 67.71 | 8.377 | 6.01 | 0.1697 |
| 0.45 (7.8 GiB) | LEDGER | 2.935 | 1.632 | 40.71 | 10.903 | 5.94 | 0.0774 |
| 0.45 | **LEDGER vs best baseline (moe-inf*)** | **1.18x** | 1.01x | 1.39x | | | |
| 0.65 (11.2 GiB) | LRU | 2.236 | 1.153 | 33.87 | 14.308 | 3.22 | 0.0571 |
| 0.65 (11.2 GiB) | MoE-Infinity* | 2.145 | 1.195 | 29.67 | 14.921 | 3.48 | 0.0430 |
| 0.65 (11.2 GiB) | Mixtral-offloading* | 2.239 | 1.152 | 33.97 | 14.292 | 3.22 | 0.0571 |
| 0.65 (11.2 GiB) | LEDGER | 1.768 | 1.100 | 20.87 | 18.100 | 2.95 | 0.0164 |
| 0.65 | **LEDGER vs best baseline (moe-inf*)** | **1.21x** | 1.09x | 1.42x | | | |

Ablation at 0.45 (qwen30b / sharegpt):

| configuration | request s | TTFT s | TPOT ms |
|---|---|---|---|
| LEDGER (full) | 2.935 | 1.632 | 40.71 |
| - async submission | 3.553 | 2.016 | 48.03 |
| - prompt routing term (history only) | 2.899 | 1.623 | 39.85 |
| - decode history term (prompt only) | 3.135 | 1.627 | 47.13 |
| - recency term | 3.243 | 1.628 | 50.48 |
| - selective admission | 2.915 | 1.630 | 40.16 |
| count utility instead of estimate | 3.079 | 1.631 | 45.22 |
| + held-out initial counts | 2.968 | 1.620 | 42.13 |
| - prefix pin | 2.904 | 1.624 | 39.99 |
| - live-set guard | 2.896 | 1.623 | 39.80 |
| pread+copy data path | 3.586 | 2.053 | 47.88 |
| no residency | 10.695 | 2.679 | 250.53 |

### qwen30b / mmlu

| budget | policy | request s | TTFT s | TPOT ms | tok/s (e2e) | prefill GiB/req | decode GiB/tok |
|---|---|---|---|---|---|---|---|
| 0.25 (4.3 GiB) | LRU | 3.357 | 2.274 | 135.39 | 2.383 | 9.97 | 0.4392 |
| 0.25 (4.3 GiB) | MoE-Infinity* | 3.369 | 2.266 | 137.83 | 2.375 | 9.97 | 0.4412 |
| 0.25 (4.3 GiB) | Mixtral-offloading* | 3.269 | 2.288 | 122.74 | 2.447 | 9.97 | 0.3911 |
| 0.25 (4.3 GiB) | LEDGER | 2.976 | 2.276 | 87.50 | 2.688 | 10.03 | 0.2369 |
| 0.25 | **LEDGER vs best baseline (mixtral*)** | **1.10x** | 1.01x | 1.40x | | | |
| 0.45 (7.8 GiB) | LRU | 2.740 | 1.696 | 130.49 | 2.920 | 6.75 | 0.4244 |
| 0.45 (7.8 GiB) | MoE-Infinity* | 1.972 | 1.709 | 32.86 | 4.057 | 6.88 | 0.0562 |
| 0.45 (7.8 GiB) | Mixtral-offloading* | 2.583 | 1.710 | 109.02 | 3.098 | 6.76 | 0.3434 |
| 0.45 (7.8 GiB) | LEDGER | 1.917 | 1.668 | 31.07 | 4.174 | 6.68 | 0.0463 |
| 0.45 | **LEDGER vs best baseline (moe-inf*)** | **1.03x** | 1.02x | 1.06x | | | |
| 0.65 (11.2 GiB) | LRU | 1.378 | 1.158 | 27.54 | 5.804 | 3.71 | 0.0361 |
| 0.65 (11.2 GiB) | MoE-Infinity* | 1.355 | 1.200 | 19.43 | 5.904 | 4.01 | 0.0165 |
| 0.65 (11.2 GiB) | Mixtral-offloading* | 1.377 | 1.157 | 27.51 | 5.811 | 3.71 | 0.0361 |
| 0.65 (11.2 GiB) | LEDGER | 1.205 | 1.075 | 16.30 | 6.640 | 3.33 | 0.0056 |
| 0.65 | **LEDGER vs best baseline (moe-inf*)** | **1.12x** | 1.12x | 1.19x | | | |

Ablation at 0.45 (qwen30b / mmlu):

| configuration | request s | TTFT s | TPOT ms |
|---|---|---|---|
| LEDGER (full) | 1.917 | 1.668 | 31.07 |
| - async submission | 2.406 | 2.121 | 35.63 |
| - prompt routing term (history only) | 1.853 | 1.658 | 24.38 |
| - decode history term (prompt only) | 2.119 | 1.685 | 54.22 |
| - recency term | 1.944 | 1.692 | 31.49 |
| - selective admission | 1.919 | 1.673 | 30.76 |
| count utility instead of estimate | 1.915 | 1.696 | 27.36 |
| + held-out initial counts | 2.106 | 1.657 | 56.10 |
| - prefix pin | 1.916 | 1.670 | 30.75 |
| - live-set guard | 1.920 | 1.675 | 30.71 |
| pread+copy data path | 2.432 | 2.148 | 35.50 |
| no residency | 4.753 | 2.773 | 247.58 |

## mixtral8x7b

### mixtral8x7b / mmlu

| budget | policy | request s | TTFT s | TPOT ms | tok/s (e2e) | prefill GiB/req | decode GiB/tok |
|---|---|---|---|---|---|---|---|

