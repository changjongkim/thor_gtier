# Jetson AGX Thor: measured platform facts

**Date:** 2026-09-22
**Device:** Jetson AGX Thor Developer Kit, module `p3834-0008`, L4T R39.2 (JetPack 7.2),
CUDA 13.0, driver 595.78, Blackwell CC 11.0 (sm_110), 20 SM, 14 ARM cores, 122 GiB,
nvpmodel MAXN, `jetson_clocks` applied.

Consolidated from the campaigns in `epoch_validation/`, `ann_coherence/` and `thor_bw/`.
These are the facts any Thor research direction has to be consistent with.

## Memory system

| | measured |
|---|---:|
| GPU STREAM-triad, mapped memory | **250.2 GB/s** |
| CPU STREAM-triad, 12 threads | **209.2 GB/s** |
| CPU + GPU concurrent | **230.3 GB/s** |
| combined / GPU alone | **0.912x** |
| board power at full bandwidth (VIN) | **67.4 W** |
| L2 | 32 MiB |
| bus / clock | 256-bit / 4266 MHz (≈273 GB/s theoretical) |

**The GPU alone reaches 91% of theoretical bandwidth, and adding the CPU makes the total
worse.** For any memory-bound workload, coherent CPU-GPU co-execution on Thor cannot add
bandwidth; it only adds contention. This kills "hybrid CPU-GPU co-execution" as a
performance strategy on this part before it is attempted.

## Capacity

Largest single `cudaHostAlloc(..., cudaHostAllocMapped)` that succeeds: **112 GiB**
(`cudaMemGetInfo` reports 104.8 GiB free of 122.8 GiB).

| representation | max state vector |
|---|---:|
| complex128 (16 B/amplitude) | **32 qubits** |
| complex64 (8 B/amplitude) | **33 qubits** |

This matches the qubit ceiling of an 80 GB A100/H100 at roughly one sixth of the power
envelope, on a single module with no PCIe staging and no multi-GPU communication.

## Coherence

First Tegra with two-way Sysmem Full Coherency: `concurrentManagedAccess`,
`pageableMemoryAccess` and `pageableMemoryAccessUsesHostPageTables` are all 1 (all 0 on
Orin).

- CPU writer on **disjoint** memory costs the GPU **+0.0%** (p = 0.55) — no bandwidth signature.
- CPU writer on the **same cache line** costs the GPU **+182.8%** (2.83x), 10/10 sessions.
- The penalty is gone at **128 B separation** and flat out to 2 KiB. It is one-cache-line
  false sharing.
- On Orin the victim was the CPU (+87.7%); on Thor it is the GPU (+182.8%). Full coherency
  moves the cost onto the accelerator.
- The effect needs a high metadata-to-payload ratio. It vanishes in payload-dominated
  workloads: zero at ANN embedding dimensions 384-1536 at every batch size.

Details: `epoch_validation/VALIDATION_THOR.md`, `ann_coherence/RESULTS.md`.

## MIG

JetPack 7.2 adds MIG on Thor: two partitions of 12 and 8 SMs (profiles 83 and 78),
consistent with the 20 SMs this device reports. Currently `MIG Mode: Disabled`; enabling
requires a reboot. NVIDIA's Jetson documentation notes that, because Thor has unified
memory, **both partitions share the same memory**. Reported preview defects include SoC
engines failing while MIG is on and `cudaMalloc` hangs.

**Untested here.** Whether the coherence effect crosses a MIG boundary is unmeasured.

## Software support gaps found

- **cuPQC 0.4.1 does not support sm_110.** `cupqc::SM<1100>` fails with
  `incomplete type "commondx::SM<1100U>" is not allowed`, while `SM<900>` instantiates.
  The precompiled libraries carry LTO code for sm_70-sm_90 only. NVIDIA's GPU PQC library
  therefore does not build for its own newest edge device.
- **CUDA 13 removed `cudaDeviceProp::memoryClockRate`**; code written for JetPack 5/6
  needs a `CUDART_VERSION` guard.
- **MPS**: `/usr/bin/nvidia-cuda-mps-control` exists and CUDA is 13.0 (past the 12.5
  Tegra-MPS threshold), but the control daemon writes its startup lines and exits
  immediately, as the user and under `sudo -E`. Unresolved.
- No torch, no VLA stack installed; anything model-based needs an sm_110 rebuild.

## What these facts rule in and out

| Idea | Status from these measurements |
|---|---|
| Hybrid CPU-GPU co-execution for bandwidth-bound work | **ruled out** — 0.912x |
| Coherence interference as a systems problem | **ruled out** — 128 B alignment fixes it |
| Coherence interference in payload-dominated domains | **ruled out** — diluted to zero |
| Large state vectors on one low-power module | **open** — 32-33 qubits at 67 W |
| Memory-hierarchy cliffs in the 29-33 qubit range | **open** — nobody has measured them |
| GPU-accelerated PQC on Blackwell edge | **open** — cuPQC does not support sm_110 |
| Coherence as a side/covert channel across MIG | **open** — MIG test not run |
