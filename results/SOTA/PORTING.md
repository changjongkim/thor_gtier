# 비교 대상 시스템의 이식 기록

## 같은 모델에서 돌릴 수 있는 것

| 시스템 | 방식 | 지원 모델 | 이식 |
|---|---|---|---|
| **Fiddler** | 전문가를 CPU에서 계산, 가중치를 옮기지 않음 | Mixtral-8x7B | 순수 PyTorch, 컴파일 없음. 2024년 핀(torch 2.1.2)을 설치하면 sm_110 PyTorch가 CPU 전용 휠로 덮이므로 `--no-deps` |
| **Mixtral-offloading** | HQQ 혼합 양자화 + LRU + 투기적 적재 | Mixtral-8x7B | HQQ 0.2.8 설치 |
| **MoE-Infinity** | EAM 기반 예측 프리페치 + 캐싱 | Mixtral, Qwen3-MoE 등 | `setup.py`가 아키텍처를 80/90/120으로 하드코딩 → CUTLASS GEMM이 `Error Internal`. `MOE_CUDA_ARCHS` 환경변수와 PTX 폴백을 넣어 재빌드 |

## 같은 모델에서 돌릴 수 없는 것

| 시스템 | 지원 모델 | 판정 |
|---|---|---|
| **Pre-gated MoE** (ISCA'24) | **Switch Transformer만** (T5 기반 인코더-디코더) | 제외 — 메커니즘은 `--lookahead`로 평가 |
| PowerInfer (SOSP'24) | ReLU-sparsified dense 모델 | 제외 — MoE 아님 |
| FlexGen (ICML'23) | OPT 계열 | 제외 — MoE 아님 |

### Pre-gated MoE를 제외한 경위

FasterTransformer(2022) 기반이라 CUDA 13 + sm_110으로 가져오는 데 다음을 차례로 거쳤다.

1. **아키텍처.** 변수는 `SM`이고, 하드코딩된 `SM_SETS`(52~90)에 없으면 실패하지 않고
   70/75/80/86 기본값으로 **조용히** 떨어진다. 110을 목록과 WMMA 활성 목록에 추가.
2. **NVTX.** CUDA 13에서 v2 헤더 `nvToolsExt.h`가 제거됨 → v3 `nvtx3/nvToolsExt.h`
   (헤더 온리, 링크 제거).
3. **C++ 표준.** CUDA 13의 libcu++가 C++17을 요구 → `-DCXX_STD=17`.
4. **전이 포함.** `printf`, `std::max`가 미정의 → `-include cstdio -include algorithm`.
5. **CCCL.** `cub::Max`, `cub::Sum`이 CUDA 13에서 제거됨.

5에서 멈췄다. 이식을 끝내도 **Switch Transformer만 돌릴 수 있고**, 그 모델은 T5 기반
인코더-디코더라 이 연구의 대상(디코더 전용 MoE LLM 서빙)과 워크로드가 다르다. 같은
모델에서 비교할 수 없는 시스템에 이식 비용을 계속 쓰는 것은 비교에 기여하지 않는다.

**메커니즘은 평가됐다.** Pre-gated의 핵심은 게이트를 한 층 앞당겨 다음 층 전문가를 미리
가져오는 것이고, 이는 드라이버의 `--lookahead`가 정확히 모델링한다. 1층부터 전층까지
쓸었을 때 토큰당 I/O는 6.80~6.92 ms로 **오차 범위**였다 — 깊은 큐와 쓸 만한 상주가
있으면 가릴 지연이 없으므로 미리 아는 것이 이득을 주지 않는다(§4.16e).

## 예산 비교에 들어오는 것과 들어오지 않는 것

| 시스템 | 전문가의 위치 | 통합 메모리에서의 발자국 | 판정 |
|---|---|---|---|
| MoE-Infinity | SSD → 호스트 캐시 → GPU | 캐시 크기로 제한 가능 (`device_memory_ratio`) | **예산 비교 대상** (bf16 트랙, `scripts/sota_serve.py`) |
| Fiddler | 모든 전문가를 호스트 메모리에, CPU에서 계산 | 모델 전체 (Mixtral bf16 87 GiB) | 예산 비교 대상 아님 — 이 풀에서 bf16 적재가 매번 OOM |
| Mixtral-offloading | 모든 전문가를 (양자화해) 호스트 메모리에, GPU 캐시로 옮김 | 양자화된 모델 전체 | 예산 비교 대상 아님 — 호스트와 GPU가 같은 풀이므로 "오프로드"가 발자국을 줄이지 않음 |

두 시스템의 전제는 GPU 메모리가 작고 호스트 메모리가 크다는 것이다. 통합 메모리에서
그 구분은 없어지므로, 이들이 하는 일은 같은 풀 안에서 바이트를 옮기는 것이 되고 모델
크기 미만의 예산에서는 실행되지 않는다. 이 연구의 설정(모델 > 예산)에서 비교할 수 있는
실제 시스템은 저장장치에서 읽는 MoE-Infinity다. 두 시스템의 메커니즘(LRU + 투기 적재)은
드라이버의 `mixtral*` 정책으로 같은 데이터 경로에서 평가한다.
