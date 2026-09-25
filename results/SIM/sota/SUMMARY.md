# 기존 시스템의 상주 정책 대비 (Qwen3-30B, 상주 35%, decode hit rate)

`scripts/sim_sota.py` (v2). LRU·LFU·PHASOR는 Fig. 9 / `sim_ablate.py`와 같은 정의
(LRU 접근 단위 시각, LFU prompt 토큰 수만큼, PHASOR 스텝 단위)이며 용량 스윕 35% 값과 0.001 이내로 일치한다.

| 워크로드 | LRU | LFU | FlashMoE* | DuoServe* (학습 MLP) | DuoServe* (pop×aff) | **PHASOR** |
|---|---:|---:|---:|---:|---:|---:|
| LongBench | 0.855 | 0.660 | 0.768 | 0.881 | 0.860 | **0.897** |
| ShareGPT | 0.839 | 0.764 | 0.823 | 0.871 | 0.846 | **0.899** |
| MMLU | 0.723 | 0.800 | 0.917 | 0.796 | 0.744 | **0.941** |

DuoServe*의 예측 품질과 비용:

| 워크로드 | 예측 top-K 정밀도 (학습 MLP / pop×aff) | 불필요 prefetch / decode 접근 (학습 MLP) |
|---|---:|---:|
| LongBench | 0.353 / 0.245 | 0.159 |
| ShareGPT | 0.444 / 0.331 | 0.128 |
| MMLU | 0.366 / 0.221 | 0.214 |

## 재현 조건

- **FlashMoE\***: 층별 슬롯 C//L = 44. 축출 FFN(1/recency, freq/max → 3층, hidden 128)은 평가하지 않는 두 워크로드의
  트레이스에서 Belady 목표로 정확히 44 슬롯에 대해 학습했다(`results/FLASHMOE/qwen30b_<w>_s44.txt`).
- **DuoServe\***: 서빙 실험과 같은 학습된 7층 MLP 예측기(`results/DUOSERVE/qwen30b_<w>.pt`, 평가하지 않는 워크로드로 학습)를 쓴다.
  층 l을 마친 뒤 층 l+1의 top-K를 예측해 prefetch하고, 복사가 제때 끝난다고 가정해 **같은 토큰의** 층 l+1에서 적중으로 센다
  (DuoServe에 유리한 가정). 바탕 캐시는 접근 단위 LRU. 입력으로 받는 pop×aff 점수만 쓴 경우를 참고로 함께 적는다.

## v1 대비 변경 (`v1_steplru_popaff_late/`)

- v1의 LRU는 스텝 단위 시각 + 인덱스 순 동점 처리였다 → 접근 단위(Fig. 9와 같음).
- v1의 DuoServe*는 pop×aff 점수를 썼고, prefetch를 적중 판정 **뒤에** 넣어 다음 토큰에만 도움이 되었다(DuoServe에 불리) →
  학습 MLP + 같은 토큰 적중. DuoServe*가 +3.7~+6.7 pp 올랐다.
- v1의 FlashMoE*는 45 슬롯으로 학습한 가중치를 44 슬롯에서 썼다 → 44 슬롯으로 학습.
