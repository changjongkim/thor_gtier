## qwen30b, budget 0.65: request time by component (s) and bytes read

| workload | system | request | prompt compute | exposed prefill I/O | decode compute | exposed decode I/O | prefill GiB/req | decode GiB/tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| longbench | LRU | 8.18 | 5.17 | 1.43 | 0.43 | 1.15 | 5.60 | 0.089 |
| longbench | MoE-Infinity* | 6.30 | 5.17 | 0.05 | 0.43 | 0.66 | 5.57 | 0.080 |
| longbench | Mixtral-offloading* | 6.41 | 5.17 | 0.04 | 0.43 | 0.77 | 5.60 | 0.089 |
| longbench | PHASOR | 5.70 | 5.17 | 0.02 | 0.43 | 0.08 | 4.88 | 0.026 |
| sharegpt | LRU | 2.97 | 0.53 | 0.97 | 0.43 | 1.04 | 3.68 | 0.082 |
| sharegpt | MoE-Infinity* | 1.81 | 0.53 | 0.51 | 0.43 | 0.34 | 3.84 | 0.051 |
| sharegpt | Mixtral-offloading* | 2.12 | 0.53 | 0.48 | 0.43 | 0.69 | 3.68 | 0.082 |
| sharegpt | PHASOR | 1.16 | 0.53 | 0.15 | 0.43 | 0.05 | 3.01 | 0.017 |
| mmlu | LRU | 1.80 | 0.45 | 1.10 | 0.11 | 0.14 | 4.17 | 0.041 |
| mmlu | MoE-Infinity* | 1.34 | 0.45 | 0.76 | 0.11 | 0.02 | 4.51 | 0.019 |
| mmlu | Mixtral-offloading* | 1.30 | 0.45 | 0.67 | 0.11 | 0.07 | 4.17 | 0.041 |
| mmlu | PHASOR | 0.83 | 0.45 | 0.27 | 0.11 | 0.00 | 3.42 | 0.006 |
