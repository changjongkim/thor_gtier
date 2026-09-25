# 데이터 경로별 CPU 비용

**측정:** `scripts/cpu_cost.sh` (SSD가 빈 구간, 파이프라인 락 보유) · 원시 로그 `CPU_COST/micro.log`, `CPU_COST/real.log`

조건은 ASYNC_FIX.md / BACKENDS.md와 같다: 32 GiB 랜덤 파일, 백엔드마다 별도 프로세스에서 페이지 캐시를 비움,
인출당 4 MiB(n = 4 MiB / read size, slot = read size), 16 GiB 구간에 흩어진 오프셋, 실행당 4 GiB, 2회 평균.

- **user / sys s/GiB**: 시간 측정 구간의 `getrusage(RUSAGE_SELF)` 증가분 ÷ 읽은 GiB. 프로세스의 모든 스레드(io_uring 작업자 포함).
- **avg cores**: (user + sys) ÷ wall.
- **machine s/GiB, machine cores**: `/proc/stat` 전체 busy 시간 증가분에서 실행 직전 3 s 유휴 기준을 뺀 값.
  GPU 페이지 폴트 처리와 완료 인터럽트는 프로세스 밖(커널 스레드, IRQ)에서 돌아 getrusage에 잡히지 않으므로 함께 적는다.

## 미시 벤치마크 (read size 스윕, 실행당 4 GiB, 2회)

실행당 4 GiB를 16 GiB 구간에서 읽으므로 mmap 계열은 readahead로 올라온 이웃 페이지를 재사용한다
(BACKENDS.md는 실행당 512 MiB). 이전 표와 같은 길이의 결과는 아래 절.

| backend | read size | GiB/s | user s/GiB | sys s/GiB | avg cores | machine s/GiB | machine cores |
|---|---:|---:|---:|---:|---:|---:|---:|
| gtier(async) | 16 KiB | 1.636 | 0.005 | 0.197 | 0.32 | 0.399 | 0.61 |
| cufile | 16 KiB | 0.539 | 0.694 | 0.999 | 0.91 | 2.318 | 1.24 |
| pread+copy | 16 KiB | 0.584 | 0.103 | 0.213 | 0.18 | 1.488 | 0.87 |
| mmap-cpu | 16 KiB | 0.785 | 0.030 | 1.549 | 1.23 | 2.370 | 1.85 |
| uvm | 16 KiB | 0.129 | 0.058 | 0.448 | 0.07 | 0.978 | 0.13 |
| mmap-gpu | 16 KiB | 0.265 | 0.010 | 0.006 | 0.00 | 2.505 | 0.66 |
| gtier(async) | 64 KiB | 3.676 | 0.002 | 0.063 | 0.24 | 0.302 | 1.11 |
| cufile | 64 KiB | 1.682 | 0.174 | 0.309 | 0.81 | 0.878 | 1.48 |
| pread+copy | 64 KiB | 1.459 | 0.040 | 0.151 | 0.28 | 0.448 | 0.65 |
| mmap-cpu | 64 KiB | 0.805 | 0.030 | 2.106 | 1.72 | 2.394 | 1.93 |
| uvm | 64 KiB | 0.352 | 0.024 | 0.331 | 0.12 | 1.129 | 0.40 |
| mmap-gpu | 64 KiB | 0.303 | 0.006 | 0.004 | 0.00 | 1.869 | 0.57 |
| gtier(async) | 256 KiB | 3.529 | 0.002 | 0.069 | 0.25 | 0.270 | 0.95 |
| cufile | 256 KiB | 2.210 | 0.064 | 0.142 | 0.46 | 0.443 | 0.98 |
| pread+copy | 256 KiB | 1.940 | 0.024 | 0.188 | 0.41 | 0.299 | 0.58 |
| mmap-cpu | 256 KiB | 1.020 | 0.021 | 2.238 | 2.30 | 2.740 | 2.79 |
| uvm | 256 KiB | 0.735 | 0.020 | 0.344 | 0.27 | 0.483 | 0.35 |
| mmap-gpu | 256 KiB | 0.476 | 0.008 | 0.007 | 0.01 | 1.378 | 0.66 |
| gtier(async) | 1 MiB | 4.176 | 0.005 | 0.067 | 0.30 | 0.186 | 0.78 |
| cufile | 1 MiB | 3.002 | 0.030 | 0.051 | 0.24 | 0.207 | 0.62 |
| pread+copy | 1 MiB | 2.749 | 0.010 | 0.154 | 0.45 | 0.264 | 0.73 |
| mmap-cpu | 1 MiB | 1.427 | 0.018 | 0.805 | 1.18 | 1.284 | 1.83 |
| uvm | 1 MiB | 1.275 | 0.013 | 0.311 | 0.41 | 0.317 | 0.40 |
| mmap-gpu | 1 MiB | 0.797 | 0.008 | 0.004 | 0.01 | 0.942 | 0.75 |
| gtier(async) | 4 MiB | 3.989 | 0.006 | 0.073 | 0.31 | 0.257 | 1.02 |
| cufile | 4 MiB | 2.456 | 0.009 | 0.079 | 0.21 | 0.222 | 0.54 |
| pread+copy | 4 MiB | 2.213 | 0.007 | 0.165 | 0.38 | 0.452 | 1.00 |
| mmap-cpu | 4 MiB | 1.271 | 0.016 | 0.316 | 0.42 | 0.648 | 0.82 |
| uvm | 4 MiB | 1.453 | 0.012 | 0.338 | 0.51 | 0.646 | 0.94 |
| mmap-gpu | 4 MiB | 1.044 | 0.003 | 0.008 | 0.01 | 0.471 | 0.52 |

## 미시 벤치마크 (BACKENDS.md와 같은 길이: 실행당 512 MiB, 3회)

| backend | read size | GiB/s | user s/GiB | sys s/GiB | avg cores | machine s/GiB | machine cores |
|---|---:|---:|---:|---:|---:|---:|---:|
| gtier(async) | 16 KiB | 1.522 | 0.004 | 0.153 | 0.24 | 0.578 | 0.86 |
| cufile | 16 KiB | 0.551 | 0.777 | 0.899 | 0.92 | 3.333 | 1.84 |
| pread+copy | 16 KiB | 0.727 | 0.143 | 0.160 | 0.22 | 0.872 | 0.63 |
| mmap-cpu | 16 KiB | 0.141 | 0.023 | 8.668 | 1.23 | 17.045 | 2.41 |
| uvm | 16 KiB | 0.122 | 0.139 | 0.579 | 0.09 | 0.715 | 0.09 |
| mmap-gpu | 16 KiB | 0.050 | 0.023 | 0.018 | 0.00 | 1.702 | 0.08 |
| gtier(async) | 64 KiB | 3.225 | 0.001 | 0.071 | 0.23 | 0.211 | 0.68 |
| cufile | 64 KiB | 1.438 | 0.187 | 0.318 | 0.73 | 1.169 | 1.68 |
| pread+copy | 64 KiB | 1.734 | 0.061 | 0.124 | 0.32 | 0.534 | 0.92 |
| mmap-cpu | 64 KiB | 0.276 | 0.023 | 5.827 | 1.62 | 7.873 | 2.17 |
| uvm | 64 KiB | 0.359 | 0.039 | 0.395 | 0.15 | 0.348 | 0.13 |
| mmap-gpu | 64 KiB | 0.113 | 0.011 | 0.013 | 0.00 | 2.913 | 0.33 |
| gtier(async) | 256 KiB | 2.212 | 0.002 | 0.078 | 0.18 | 0.084 | 0.19 |
| cufile | 256 KiB | 1.560 | 0.073 | 0.159 | 0.36 | 0.443 | 0.69 |
| pread+copy | 256 KiB | 1.542 | 0.029 | 0.189 | 0.34 | 0.239 | 0.37 |
| mmap-cpu | 256 KiB | 0.633 | 0.036 | 3.438 | 2.20 | 4.239 | 2.68 |
| uvm | 256 KiB | 0.669 | 0.011 | 0.385 | 0.27 | 0.872 | 0.58 |
| mmap-gpu | 256 KiB | 0.257 | 0.013 | 0.007 | 0.01 | 2.449 | 0.63 |
| gtier(async) | 1 MiB | 3.578 | 0.003 | 0.070 | 0.26 | 0.103 | 0.38 |
| cufile | 1 MiB | 2.628 | 0.035 | 0.054 | 0.23 | 0.232 | 0.61 |
| pread+copy | 1 MiB | 2.562 | 0.025 | 0.125 | 0.38 | 0.354 | 0.91 |
| mmap-cpu | 1 MiB | 0.970 | 0.017 | 1.302 | 1.28 | 2.061 | 2.00 |
| uvm | 1 MiB | 1.101 | 0.021 | 0.380 | 0.44 | 0.197 | 0.22 |
| mmap-gpu | 1 MiB | 0.550 | 0.006 | 0.006 | 0.01 | 1.398 | 0.77 |
| gtier(async) | 4 MiB | 4.286 | 0.002 | 0.057 | 0.25 | 0.118 | 0.51 |
| cufile | 4 MiB | 1.966 | 0.024 | 0.039 | 0.12 | 0.183 | 0.36 |
| pread+copy | 4 MiB | 2.246 | 0.010 | 0.163 | 0.39 | 0.274 | 0.61 |
| mmap-cpu | 4 MiB | 0.986 | 0.019 | 0.399 | 0.41 | 0.711 | 0.70 |
| uvm | 4 MiB | 1.354 | 0.007 | 0.411 | 0.57 | 0.400 | 0.54 |
| mmap-gpu | 4 MiB | 0.876 | 0.003 | 0.008 | 0.01 | 1.064 | 0.93 |

## 실제 모델 설정 (Qwen3-30B-A3B bf16, 128 중 8 전문가, skew 0.8, batch 8, window 2 GiB = 512 × 4 MiB, cgroup 8 GiB)

| backend | read size | GiB/s | user s/GiB | sys s/GiB | avg cores | machine s/GiB | machine cores |
|---|---:|---:|---:|---:|---:|---:|---:|
| gtier(async) | 4 MiB | 4.502 | 0.002 | 0.061 | 0.28 | 0.128 | 0.57 |
| cufile | 4 MiB | 3.684 | 0.011 | 0.051 | 0.23 | 0.200 | 0.74 |
| pread+copy | 4 MiB | 3.460 | 0.005 | 0.113 | 0.41 | 0.183 | 0.63 |
| mmap-cpu | 4 MiB | 0.339 | 0.017 | 16.520 | 5.61 | 16.790 | 5.70 |
| uvm | 4 MiB | 1.564 | 0.005 | 0.422 | 0.67 | 0.587 | 0.92 |
| mmap-gpu | 4 MiB | 2.040 | 0.002 | 0.001 | 0.01 | 0.330 | 0.67 |

## 참고

- cufile was OOM-killed under the 8 GiB cap (footprint 10.6 GiB, HF_MOE/FOOTPRINT.md); rerun under a 16 GiB cap
- excluded (disturbed by a drop_caches during the run): micro_iters128.log RUN rep=1 item=1048576 backend=0
- excluded (disturbed by a drop_caches during the run): micro_iters128.log RUN rep=1 item=1048576 backend=4

## 실패한 실행

- `RUN backend=4 FAILED rc=137`
