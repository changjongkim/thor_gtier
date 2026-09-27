# ZipMoE fidelity on its own model and harness (paper 4.1)

Model: Qwen1.5-MoE-A2.7B-Chat (the ZipMoE paper's model). Harness: ZipMoE's `evaluation/evaluate.py`,
24 ShareGPT prompts (its sampler, seed 321), 512-token prompts, 64 new tokens, 30 s cooling per prompt,
after its warm-up prompt. Footprint presets are fractions of a 64 GB Orin; scaled by 64/122.8 so a
footprint means the same bytes here (`ZIPMOE_MEM_SCALE`). No page-cache scrubbing (its native setting).

## (a) Its caching against its LRU / LFU options (mean end-to-end s per request)

| footprint | ZipMoE | LRU | LFU | ZipMoE vs LRU | ZipMoE vs LFU |
|---:|---:|---:|---:|---:|---:|
| 10 GB | 23.76 (24) | 21.58 (24) | 22.44 (24) | +10.1% | +5.9% |
| 20 GB | 16.91 (24) | 16.50 (24) | 17.18 (24) | +2.5% | -1.6% |
| 30 GB | 9.84 (24) | 9.43 (24) | 8.96 (24) | +4.4% | +9.9% |

## (b) Our runner against its harness, same prompts and memory (20 GB)

| | end-to-end s | TTFT s | TPOT ms |
|---|---:|---:|---:|
| ZipMoE harness | 16.91 | 1.43 | 246 |
| our runner | 17.09 | 1.57 | 246 |
| our runner, first request excluded (its harness warms up) | 16.77 | | |

Difference (warm): -0.8%.
