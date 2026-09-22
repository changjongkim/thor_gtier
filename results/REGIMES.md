# GPU 파일 백업 mmap regime 원측정값

**장비:** Jetson AGX Thor, JetPack 7.2, CUDA 13.0, sm_110, 122.8 GiB, `jetson_clocks` 적용
**측정일:** 2026-09-22
**프로브:** [`../mmap_probe/mmaptest.cu`](../mmap_probe/mmaptest.cu)

## 방법

`ftruncate`로 파일을 만들고 `MAP_SHARED`로 `mmap`한 뒤, GPU 커널이 **4 KiB 페이지당 하나의
`double`** 을 읽는다. 즉 순수하게 페이지 폴트 동작만 측정한다. 실효 대역폭은
(파일 크기 / 커널 실행 시간)으로, 폴트 서비스 경로가 낼 수 있는 유효 처리량이다.

## 결과

| 파일 크기 | 결과 | 시간 | 실효 대역폭 | NVMe(4.9 GB/s) 대비 |
|---:|---|---:|---:|---:|
| 8 GiB | OK | 5.93 s | 1.35 GiB/s | 28% |
| 12 GiB | OK | 9.05 s | 1.33 GiB/s | 27% |
| 16 GiB | OK | 12.43 s | 1.29 GiB/s | 26% |
| 20 GiB | OK | 55.49 s | 0.36 GiB/s | 7% |
| 24 GiB | OK | 106.06 s | 0.23 GiB/s | 5% |
| 32 GiB | OK | 122.38 s | 0.26 GiB/s | 5% |
| 48 GiB | OK | 227.93 s | 0.21 GiB/s | 4% |
| 64 GiB | **CRASH** | — | — | — |
| 80 GiB | OK | 411.91 s | 0.19 GiB/s | 4% |
| 100 GiB | **CRASH** | 46.6 s 후 | — | — |
| 200 GiB | **CRASH** | 348.5 s 후 | — | — |

## 관찰

- **16→20 GiB 절벽.** 1.29 → 0.36 GiB/s, **3.6배**. 이후 0.19~0.26 GiB/s에서 평탄.
- **천장.** 가장 빠른 구간조차 장치 순차 읽기(4.9 GB/s)의 28%에 불과하다.
- **비결정성.** 64 GiB는 실패하고 80 GiB는 성공한다. 크기 임계값이 아니다.
  100 GiB 실패 시점에 `free`는 102 GiB가 남아 있었고 `buff/cache`는 16 GiB뿐이었다 —
  **메모리 압력 때문이 아니다.** reclaim / SMMU invalidation 경쟁으로 보인다.
- **폴백 없음.** 이 장비의 swap은 0 B다.

## 미해결

- 붕괴의 정확한 원인 — SMMU/ATS invalidation인지, 커널 reclaim 경로인지, CUDA 폴트 핸들러인지
- `mlock` / `MAP_POPULATE` / `madvise`가 붕괴를 제거하는지
- 절벽의 미세아키텍처 원인 (GPU 페이지테이블 캐시 용량 추정)
