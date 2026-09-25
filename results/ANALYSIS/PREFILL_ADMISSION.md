# Prefill admission: why PHASOR's decode lost to FlashMoE* at 0.25, and the fix

E1 (Qwen3-30B, MMLU, budget 0.25): PHASOR TPOT 568 ms vs FlashMoE* 355 ms, PHASOR flat across
requests while FlashMoE* fell from 883 to 258 ms as its cache warmed.

Diagnosis (`tpot_diag/`, 8 MMLU requests, same cap, stage 5 paused between runs):
decode hit PHASOR 0.688 vs FlashMoE* 0.740; per decode step PHASOR waited 200 ms on 120 misses
(1.05 GiB at the device limit). Mapped-memory reads were ruled out (`MEMTYPE.md`: 1.6 ms/step).
FlashMoE* never lets a prefill-staged expert displace a cached one; PHASOR admitted prefill units by
value, and a prompt's routing union (78% of units on MMLU) with the prompt term pushed out the units
decode keeps using, every request.

Fix: a prefill-staged unit enters only a free slot (decode admission unchanged). Same 8 requests:

| | request s | TTFT s | TPOT ms | decode hit | prefill hit | peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| MMLU, admit by value | 15.14 | 11.08 | 581 | 0.688 | 0.067 | 18.95 |
| MMLU, free slots only | 10.30 | 8.08 | 317 | 0.866 | 0.287 | 19.26 |
| ShareGPT, admit by value | 24.39 | 10.30 | 455 | 0.779 | 0.070 | 19.17 |
| ShareGPT, free slots only | 21.02 | 7.56 | 434 | 0.798 | 0.289 | 19.16 |

Adopted as PHASOR's default; the old behaviour is E7's `pfall` ablation. The trace simulator admitted
prefill the same way and did not show the loss, because it has no staging window: in the real engine the
prompt's non-resident units are read anyway and only their admission is optional.
Peaks moved by +1.6% at most, so the memcal targets of the baselines stand. The PHASOR E1 cells measured
before the change are in `results/MATRIX5/qwen30b/archive_prefill_admit_by_value/`; only PHASOR is rerun.
