# llama.cpp (stock) on the unified-memory Thor

- Code: llama.cpp commit 6f767fe96 (2026-09-28), llama-server built with CUDA for sm_110.
- mixtral8x7b: smoke failed: HOSTGUARD kill: MemAvailable 11756 MiB < 12 GiB (cgroup charges do not include cudaMalloc) (results/PREP/llamacpp/smoke_mixtral8x7b.log)
- Code: llama.cpp commit 6f767fe96 (2026-09-28), llama-server built with CUDA for sm_110.
- Code: llama.cpp commit 6f767fe96 (2026-09-28), llama-server built with CUDA for sm_110.
- Code: llama.cpp commit 6f767fe96 (2026-09-28), llama-server built with CUDA for sm_110.
- Qwen3-30B GGUF: unsloth/Qwen3-30B-A3B-GGUF BF16 (llama.cpp's converter needs > 110 GiB for this checkpoint's 18,867 per-expert tensors; the public conversion of the same weights is used, checked against stock tokens)
- Tokens vs stock transformers (2 prompts x 24, greedy): identical
- Code: llama.cpp commit 6f767fe96 (2026-09-28), llama-server built with CUDA for sm_110.
- Tokens vs stock transformers (2 prompts x 24, greedy): identical
- Code: llama.cpp commit 6f767fe96 (2026-09-28), llama-server built with CUDA for sm_110.
- Code: llama.cpp commit 6f767fe96 (2026-09-28), llama-server built with CUDA for sm_110.

- 09-29: dropped from the comparison (user): too slow to be a meaningful comparison. The one valid Mixtral request (20 GiB cap, before the GPU was lost at 09-28 23:52): 854 s (TTFT 640 s, TPOT 30.6 s), results/PREP/llamacpp/smoke_mixtral8x7b_first.log. Qwen3 runs after 23:52 used no GPU: archived.
