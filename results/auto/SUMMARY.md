# 자동 실험 큐 요약

생성: 2026-09-23 02:32

유효 대역폭 = 요청한 바이트 ÷ 경과 시간. tok/s = 모델 1회 통과를 1 토큰으로 센 값.


## A. Dense 모델 (Qwen2.5-32B Q8_0, 32.42 GiB — DRAM에 들어감)

토큰 1회 = 모델 전체 1회 통과. 콜드 캐시.

| 실행 | 백엔드 | GiB/s | tok/s |
|---|---|---:|---:|
| a_dense_gtier_async | gtier | 5.411 | 0.1669 |
| a_dense_backend0 | gtier | 4.911 | 0.1515 |
| a_dense_backend4 | cufile | 4.324 | 0.1333 |
| a_dense_backend3 | pread+copy | 3.992 | 0.1231 |
| a_dense_backend2 | mmap-cpu | 3.784 | 0.1167 |
| a_dense_backend5 | uvm | 2.716 | 0.0838 |
| a_dense_backend1 | mmap-gpu | 1.445 | 0.0446 |

## B. DRAM 초과 모델 (Qwen3-235B-A22B Q4_K_M, 132.4 GiB vs DRAM 122.8 GiB)

이 모델은 DRAM에 들어가지 않는다.

| 실행 | 백엔드 | GiB/s | tok/s |
|---|---|---:|---:|
| b_moe_gtier_async | gtier | 5.516 | 0.0417 |
| b_moe_backend0 | gtier | 5.182 | 0.0391 |
| b_moe_backend4 | cufile | 4.917 | 0.0371 |
| b_moe_backend3 | pread+copy | 4.450 | 0.0336 |
| b_moe_backend5 | uvm | 2.642 | 0.0200 |
| b_moe_backend1 | mmap-gpu | 0.641 | 0.0048 |
| b_moe_backend2 | mmap-cpu | 0.567 | 0.0043 |

## C. 상주 정책 — PIN vs LRU (MoE 132.4 GiB, 토큰 3회)

순환 스캔에서 LRU는 윈도우가 워킹셋을 덮기 전까지 이득이 없고, PIN은 비례한다.

| 윈도우 GiB | 모델 대비 | 정책 | GiB/s | tok/s | 히트율 | 증폭 |
|---:|---:|---|---:|---:|---:|---:|
| 12.0 | 9.1% | BLOCK(LRU) | 5.134 | 0.0388 | 51.8% | 1.00x |
| 12.0 | 9.1% | PIN | 5.583 | 0.0422 | 7.1% | 0.94x |
| 24.0 | 18.1% | BLOCK(LRU) | 5.031 | 0.0380 | 51.8% | 1.00x |
| 24.0 | 18.1% | PIN | 6.014 | 0.0454 | 14.2% | 0.89x |
| 48.0 | 36.3% | BLOCK(LRU) | 5.026 | 0.0380 | 51.8% | 1.00x |
| 48.0 | 36.3% | PIN | 6.575 | 0.0497 | 28.5% | 0.77x |
| 72.0 | 54.4% | BLOCK(LRU) | 5.014 | 0.0379 | 51.8% | 1.00x |
| 72.0 | 54.4% | PIN | 7.573 | 0.0572 | 42.8% | 0.66x |
| 96.0 | 72.5% | BLOCK(LRU) | 5.009 | 0.0378 | 51.8% | 1.00x |
| 96.0 | 72.5% | PIN | 9.118 | 0.0689 | 57.1% | 0.55x |

## D. 실제 MoE 트레이스 입도별

| 실행 | 백엔드 | GiB/s | tok/s |
|---|---|---:|---:|
| d_moe_gran_1048576_b0 | gtier | 5.277 | 0.0399 |
| d_moe_gran_1048576_b1 | mmap-gpu | 0.637 | 0.0048 |
| d_moe_gran_1048576_b4 | cufile | 4.987 | 0.0377 |
| d_moe_gran_262144_b0 | gtier | 5.059 | 0.0382 |
| d_moe_gran_262144_b1 | mmap-gpu | 0.703 | 0.0053 |
| d_moe_gran_262144_b4 | cufile | 4.338 | 0.0328 |
| d_moe_gran_65536_b0 | gtier | 4.587 | 0.0346 |
| d_moe_gran_65536_b1 | mmap-gpu | 0.626 | 0.0047 |
| d_moe_gran_65536_b4 | cufile | 3.630 | 0.0274 |
