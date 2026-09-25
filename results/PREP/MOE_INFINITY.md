# MoE-Infinity on the unified-memory Thor: cannot run within any evaluated budget

- Code: its release (`/home/thor/kcj/MoE-Infinity`, t26 venv), run unmodified through `scripts/sota_serve.py`.
- Design: at load it copies **every expert** from its offload store into a host memory pool
  (`core/model/model_topology.cpp:683`, `kHostMemoryPool->AllocateMemory` per sparse node), reading each
  partition file whole into a temporary buffer first (`read_partition`, line 640ff). There is no knob for
  the host pool; `device_memory_ratio` sizes only its GPU cache (cudaMalloc), which on this SoC is the same pool.
- Qwen3-30B-A3B bf16 (57 GiB):
  - `--budget-gib 25.65` (GPU cache 25.65 GiB, stage 4 smoke, no guard then): no request in 95 min, host at
    10 GiB available with kcompactd at 100% — stopped by hand.
  - `--budget-gib 4` (GPU cache 4 GiB, stage 5 smoke): killed by the host guard during the expert load
    ("TOPO: 99 stages, 48 sparse" was its last output) at 11.4 GiB available, i.e. about 111 GiB in use.
- Its load peak is therefore above every budget in the plan (largest: 1.08 x 57 = 61.6 GiB) by the
  MemAvailable criterion that applies to every system, so it is reported as **cannot run within budget**
  at all budgets, with this evidence, rather than tuned or modified.
- Logs: `results/PREP/smoke_mi_qwen30b.log`, `results/MATRIX4/qwen30b/` stage 4 smoke, `results/PIPELINE/pipeline.log`.
