## qwen30b, budget 0.25: request time by component (s) and bytes read

| workload | system | request | prompt compute | exposed prefill I/O | decode compute | exposed decode I/O | prefill GiB/req | decode GiB/tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| longbench | LRU | 13.51 | 5.17 | 3.24 | 0.43 | 4.67 | 11.65 | 0.348 |
| longbench | MoE-Infinity* | 10.62 | 5.17 | 0.30 | 0.43 | 4.71 | 11.64 | 0.384 |
| longbench | Mixtral-offloading* | 10.17 | 5.17 | 0.31 | 0.43 | 4.26 | 11.65 | 0.348 |
| longbench | PHASOR | 8.03 | 5.17 | 0.16 | 0.43 | 2.26 | 11.30 | 0.281 |
| sharegpt | LRU | 8.00 | 0.53 | 2.46 | 0.43 | 4.59 | 9.73 | 0.410 |
| sharegpt | MoE-Infinity* | 7.29 | 0.53 | 1.93 | 0.43 | 4.40 | 9.73 | 0.438 |
| sharegpt | Mixtral-offloading* | 6.96 | 0.53 | 1.92 | 0.43 | 4.08 | 9.73 | 0.407 |
| sharegpt | PHASOR | 4.58 | 0.53 | 1.20 | 0.43 | 2.42 | 9.32 | 0.330 |
| mmlu | LRU | 4.44 | 0.45 | 2.66 | 0.11 | 1.22 | 10.54 | 0.450 |
| mmlu | MoE-Infinity* | 3.91 | 0.45 | 2.21 | 0.11 | 1.14 | 10.55 | 0.453 |
| mmlu | Mixtral-offloading* | 3.87 | 0.45 | 2.20 | 0.11 | 1.10 | 10.54 | 0.446 |
| mmlu | PHASOR | 2.47 | 0.45 | 1.46 | 0.11 | 0.45 | 10.12 | 0.247 |
