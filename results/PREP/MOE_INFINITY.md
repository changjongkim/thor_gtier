# MoE-Infinity on the unified-memory Thor: cannot run within any evaluated budget

Version note (09-28): the result below is for its current release (96695c9, 2026-09-23). Since c098c15
(2026-02-16, "Upstream (#72)") it copies every expert into its host pool at load. Its last release with an
SSD tier, 48bb3bc (2025-02-13), kept experts in the offload store and moved them SSD -> host pool -> GPU on
demand -- the mode ZipMoE (ICML'26) compared against. That release supports Mixtral but not Qwen3; stage 12
(`scripts/stage12_moeinf_legacy.sh`) runs it in the Mixtral MMLU generality check, and its outcome is
appended below.

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

## Version 48bb3bc (2025-02-13): the last release with an SSD tier
- Experts stay in its offload store on the SSD and move SSD -> host pool -> GPU on demand; the preloading
  ("Moving sparse parameters to CPU") arrived in c098c15 (2026-02-16). Supports Mixtral, not Qwen3.
- Build: its C++ ops do not compile with this host's toolchain (GCC 13.3, libstdc++ 13):
  `core/aio/archer_prio_aio_handle.cpp:18` -- "partial specialization of struct std::hash<std::string> after
  instantiation" and the errors that follow (`results/PREP/moeinf2502/build.log`), so the 2025-02 release was
  never installed. (Stage 12's smoke then imported the current release, which the venv also sees, and failed
  on its transformers pin; that ImportError is a consequence, not the cause.)
- Build fixed 09-28 22:40 with build-compatibility changes only (`-include string`; log constant
  kMaxNumericSize 32 -> 48 for aarch64's 128-bit long double; `third_party/moeinf2502_build.patch`); it builds
  and imports. Its run (stage 12 again, after stage 11) replaces this entry:
- (superseded) the 2025-02 SSD-tier release cannot be built here unmodified; not run. Its current release preloads
  every expert (above). Recorded as cannot run on Mixtral (`results/MATRIX5/mixtral8x7b/mmlu/moeinf2502_*.txt`).

## Version 48bb3bc (2025-02-13): the last release with an SSD tier
- Experts stay in its offload store on the SSD and move SSD -> host pool -> GPU on demand; the preloading
  ("Moving sparse parameters to CPU") arrived in c098c15 (2026-02-16). Supports Mixtral, not Qwen3.
- Built in its own venv (transformers 4.46.3, < 4.47 as it requires) with build-compatibility fixes only:
  `-include string` (GCC 13) and its log buffer constant kMaxNumericSize 32 -> 48 (aarch64 long double;
  third_party/moeinf2502_build.patch). Caching, prefetching and the data path are unchanged. HOST_MEMORY_RATIO (its build-time
  host pool size, default 0.8 of system memory) = 0.04; memcal calibrates device_memory_ratio.

## Version 48bb3bc (2025-02-13): the last release with an SSD tier
- Experts stay in its offload store on the SSD and move SSD -> host pool -> GPU on demand; the preloading
  ("Moving sparse parameters to CPU") arrived in c098c15 (2026-02-16). Supports Mixtral, not Qwen3.
- Built in its own venv (transformers 4.46.3, < 4.47 as it requires) with build-compatibility fixes only:
  `-include string` (GCC 13) and its log buffer constant kMaxNumericSize 32 -> 48 (aarch64 long double;
  third_party/moeinf2502_build.patch). Caching, prefetching and the data path are unchanged. HOST_MEMORY_RATIO (its build-time
  host pool size, default 0.8 of system memory) = 0.04; memcal calibrates device_memory_ratio.

## Version 48bb3bc (2025-02, SSD tier), Mixtral-8x7B: does not serve: KeyError: (0, tensor([[5, 1],

## 48bb3bc run (09-28 23:01): fails at its first request

Built with the compatibility fixes above, it loaded Mixtral and built its offload store, then stopped at
the first request: `moe_infinity/models/mixtral.py:71` passes the routed expert tensor itself to
`fetch_experts_lock_cache`, which uses each row as a dict key (`KeyError: (0, tensor([[5, 1], ...]))`,
`results/PREP/moeinf2502/smoke.log`). 48bb3bc is a dev-branch merge; the release before it, 350f0dd
(2024-08-15), has the predictor + prefetch path there and the same SSD tier. Stage 12b runs 350f0dd after
stage 11q (`scripts/stage12b_moeinf_legacy.sh`).

## Version 350f0dd (2024-08-15): release with an SSD tier (48bb3bc, 2025-02, fails at its first Mixtral request)
- Experts stay in its offload store on the SSD and move SSD -> host pool -> GPU on demand; the preloading
  ("Moving sparse parameters to CPU") arrived in c098c15 (2026-02-16). Supports Mixtral, not Qwen3.
- Built in its own venv (transformers 4.46.3, < 4.47 as it requires) with build-compatibility fixes only:
  `-include string` (GCC 13) and its log buffer constant kMaxNumericSize 32 -> 48 (aarch64 long double;
  third_party/moeinf2502_build.patch, where it applies). Caching, prefetching and the data path are unchanged. HOST_MEMORY_RATIO (its build-time
  host pool size, default 0.8 of system memory) = 0.04; memcal calibrates device_memory_ratio.

(09-29 02:42: the 350f0dd attempt met a GPU that the driver had lost at 23:52 -- see pipeline.log; not a result, rerun after the GPU is back.)

## Version 350f0dd (2024-08-15): release with an SSD tier (48bb3bc, 2025-02, fails at its first Mixtral request)
- Experts stay in its offload store on the SSD and move SSD -> host pool -> GPU on demand; the preloading
  ("Moving sparse parameters to CPU") arrived in c098c15 (2026-02-16). Supports Mixtral, not Qwen3.
- Built in its own venv (transformers 4.46.3, < 4.47 as it requires) with build-compatibility fixes only:
  `-include string` (GCC 13) and its log buffer constant kMaxNumericSize 32 -> 48 (aarch64 long double;
  third_party/moeinf2502_build.patch, where it applies). Caching, prefetching and the data path are unchanged. HOST_MEMORY_RATIO (its build-time
  host pool size, default 0.8 of system memory) = 0.04; memcal calibrates device_memory_ratio.

## Version 350f0dd (2024-08, SSD tier), Mixtral-8x7B: does not serve: 
