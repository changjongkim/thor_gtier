## qwen30b, budget 0.45: request time by component (s) and bytes read

| workload | system | request | prompt compute | exposed prefill I/O | decode compute | exposed decode I/O | prefill GiB/req | decode GiB/tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| longbench | LRU | 10.05 | 5.17 | 2.33 | 0.43 | 2.12 | 8.50 | 0.157 |
| longbench | MoE-Infinity* | 7.16 | 5.17 | 0.13 | 0.43 | 1.43 | 8.51 | 0.148 |
| longbench | Mixtral-offloading* | 7.21 | 5.17 | 0.13 | 0.43 | 1.47 | 8.51 | 0.155 |
| longbench | PHASOR | 6.15 | 5.17 | 0.04 | 0.43 | 0.50 | 8.06 | 0.089 |
| sharegpt | LRU | 4.93 | 0.53 | 1.67 | 0.43 | 2.31 | 6.55 | 0.195 |
| sharegpt | MoE-Infinity* | 3.75 | 0.53 | 1.17 | 0.43 | 1.62 | 6.59 | 0.165 |
| sharegpt | Mixtral-offloading* | 3.87 | 0.53 | 1.16 | 0.43 | 1.75 | 6.55 | 0.179 |
| sharegpt | PHASOR | 2.02 | 0.53 | 0.62 | 0.43 | 0.44 | 6.03 | 0.081 |
| mmlu | LRU | 3.59 | 0.45 | 1.86 | 0.11 | 1.17 | 7.29 | 0.432 |
| mmlu | MoE-Infinity* | 2.13 | 0.45 | 1.45 | 0.11 | 0.12 | 7.40 | 0.065 |
| mmlu | Mixtral-offloading* | 2.84 | 0.45 | 1.43 | 0.11 | 0.85 | 7.30 | 0.351 |
| mmlu | PHASOR | 1.47 | 0.45 | 0.85 | 0.11 | 0.06 | 6.77 | 0.048 |
