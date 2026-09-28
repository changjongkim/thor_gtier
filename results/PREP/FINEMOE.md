# FineMoE (EuroSys'26) on the unified-memory Thor

- Code: its release (github.com/IntelliSys-Lab/FineMoE-EuroSys26, commit 80717e9) with the Qwen3-MoE port
  (`third_party/finemoe_qwen3_sm110.patch`: attention graphs without the Qwen3.5 output gate or linear
  attention, no shared expert, model-type checks); run through `scripts/finemoe_serve.py`.
- Design: the whole checkpoint is loaded to CPU (`from_pretrained(device_map="cpu")`), then every expert
  is copied into one pinned host buffer (`model_offload.py:72`, `torch.empty(..., pin_memory=True)`):
  54.0 GiB for Qwen3-30B-A3B, before the GPU expert cache (`cache_size` slots). Its README asks
  for 192 GB of host memory. On this SoC host and GPU memory are one 122.8 GiB pool.

## Result: cannot serve on this device

- Load test (2 MMLU prompts, 4 GiB GPU cache, host guard at 12 GiB available): HOSTGUARD kill: MemAvailable 12264 MiB < 12 GiB (cgroup charges do not include cudaMalloc)
- Loading needs the CPU copy of the checkpoint and the pinned expert buffer at once, above what the
  pool holds beside the OS; FineMoE is reported as cannot run within any budget here, with this
  evidence, rather than modified.

Log: `results/PREP/finemoe/smoke.log`.
