# 실험 계획

실행 스크립트: `scripts/pipeline.sh`(라우팅 캡처) → `scripts/stage2.sh`(서빙 행렬).
결과: `results/MATRIX/<model>/<workload>/<run>.txt`, 요약 `results/MATRIX/SUMMARY.md`.

## 0. 무엇이 LEDGER인가

LEDGER는 **모든 컴포넌트가 켜진 하나의 설정**이다. 스위치는 기여 분해용이다.

```
층 1 — gTier 데이터 경로
   복사 0회 스테이징        cudaHostAlloc(Mapped), 착륙지가 곧 GPU 포인터
   정확 인출                증폭 없음, 병합 없음
   연속 제출                gtier_submit / gtier_wait  (기본 ON)

층 2 — 효용 기반 상주
   U(u) = h_prefill(u) + W · h_decode(u)     관측으로 누적
   승인은 수요가 결정        접근되는 것은 막지 않는다
   축출은 효용이 결정        현재 토큰이 쓰는 것과 핀된 프리픽스는 제외
   프리픽스 식별            온라인, 라우팅 일치율 >= 0.90, 패밀리 LRU 축출
   초기 카운트              같은 모델의 다른 워크로드 트레이스 (평가 대상 트레이스는 쓰지 않음)
```

## 1. 공통 조건

| 항목 | 값 |
|---|---|
| 모델 | Qwen3-30B-A3B, Mixtral-8x7B, Qwen3-235B-A22B — 모두 Q4_K_M GGUF, 같은 I/O 경로 |
| 워크로드 | LongBench(21), ShareGPT(24, 다중 턴 11), MMLU 5-shot(24). **모든 모델에 같은 프롬프트** (235B는 워크로드당 앞 8개) |
| 프롬프트/생성 | 최대 8192 토큰, 32 토큰 생성 (라우팅은 실제 모델 실행에서 캡처) |
| 예산 | 모델 바이트의 0.25 / 0.45 / 0.65 |
| 윈도우 | 0.5 GiB |
| 반복 | 워크로드 2회 순회 (첫 순회는 콜드) |
| 연산 | llama-bench로 같은 GGUF를 전 층 GPU에 올려 측정한 토큰당 비용(디코드/프롬프트). 235B는 들어가지 않으므로 같은 아키텍처인 30B의 측정값에 토큰당 접근 바이트 비를 곱함 (`results/MATRIX/calib.tsv`) |
| 격리 | 한 번에 한 실행, 실행 전 페이지 캐시 비움, 모든 실행은 memguard 통과 |

**주 지표**: 요청 총 시간 = 프리필 I/O + 프롬프트 연산 + Σ(디코드 I/O + 디코드 연산).
**부 지표**: TTFT, TPOT, end-to-end tok/s, 요청당 프리필 바이트, 토큰당 디코드 바이트.

## 2. E1 — 주 결과 (모델 × 워크로드 × 예산)

| 분류 | 정책 | 대응 |
|---|---|---|
| 기본 | `lru` | 수요 적재 + LRU |
| SOTA 재현 | `moe-inf*` | MoE-Infinity — 시퀀스 활성 집합 보존 |
| SOTA 재현 | `mixtral*` | Mixtral-offloading — LRU + 직전 토큰 기반 투기 적재 |
| **본 연구** | `ledger` | 전 컴포넌트 ON |

Pre-gated MoE의 메커니즘(게이트 선행)은 `--lookahead`로 평가됐다(README §4.16e).

## 3. E2 — 어블레이션 (예산 0.45)

연속 제출 off · 초기 카운트 off · 초기 카운트 ×0.25 · 프리픽스 핀 off · 현재 토큰 보호 off ·
`W` ∈ {1, 4, 16, 측정 비} · 데이터 경로 pread+copy · 상주 없음. 30B와 Mixtral은 세
워크로드, 235B는 LongBench.

## 4. E3 — 실제 시스템 (bf16 트랙)

MoE-Infinity는 HF 체크포인트만 읽으므로 Qwen3-30B bf16에서 같은 프롬프트, 같은 절대
예산으로 돌리고, 드라이버도 같은 bf16 파일을 실제 커널(`--compute`)로 돌린다.
Fiddler와 Mixtral-offloading은 모든 전문가를 호스트 메모리에 두는 설계라 통합 메모리
풀에서는 모델 크기 미만의 예산에서 실행할 수 없다 — 예산 비교 대상이 아님을 기록한다
(`results/SOTA/PORTING.md`).

## 5. 외부 기준점

llama.cpp on 235B (mmap, ngl 0): TTFT 36.6 s, TPOT 908 ms, 238.1 GiB 읽기(README §4.16d).
`-ngl 99`는 메모리 초과로 호스트를 재부팅시켰으므로 memguard가 거부한다.
