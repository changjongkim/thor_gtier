# gTier — Coherent 엣지 SoC의 Out-of-Core LLM 추론 계층

> **한 문장.** 메모리를 초과하는 모델을 돌리는 모든 시스템은 *GPU 메모리 / 호스트 DRAM / 스토리지* 의
> **3계층**을 전제하지만, coherent 엣지 SoC에서는 1·2계층이 **물리적으로 같은 메모리**다. 설계 공간이
> **DRAM↔플래시 단일 경계**로 붕괴하고, 하드웨어가 새로 내놓은 답(호스트 페이지테이블 기반 GPU demand
> paging)은 **장치 대역폭의 4~28%만 내면서 비결정적으로 죽는다.** gTier는 붕괴한 계층 구조가 실제로
> 필요로 하는 계층을 만든다.

**플랫폼:** NVIDIA Jetson AGX Thor (JetPack 7.2 / L4T R39.2, CUDA 13.0, sm_110, Blackwell CC 11.0)
· 122.8 GiB 통합 coherent 메모리 · WD SN5000S 1 TB NVMe · **swap 없음**

---

## 1. 배경 — Thor에서 실측한 사실

이 저장소의 모든 주장은 이 장비에서 직접 측정한 값에 기반한다. 전체는 [`docs/PLATFORM_NOTES.md`](docs/PLATFORM_NOTES.md).

| 항목 | 측정값 |
|---|---:|
| GPU STREAM-triad | **250.2 GB/s** (이론치 273 GB/s의 91%) |
| CPU STREAM-triad (12스레드) | 209.2 GB/s |
| **CPU + GPU 동시** | **230.3 GB/s (0.912x)** |
| 풀 대역폭시 보드 전력 (VIN) | 67.4 W |
| 최대 단일 매핑 할당 | 112 GiB |
| NVMe 순차 읽기 (O_DIRECT) | 4.9 GB/s |
| **DRAM : NVMe 대역폭 비** | **71 : 1** |
| Swap | **0 B** |

두 가지가 즉시 따라 나온다.

1. **GPU 혼자 메모리 컨트롤러를 포화시킨다.** CPU를 더하면 총합이 *줄어든다*. 메모리 바운드
   워크로드에서 coherent CPU-GPU 협업은 대역폭을 늘리지 못하고 경합만 추가한다.
2. **폴백이 없다.** swap이 0이므로 작업집합이 DRAM을 넘으면 우아한 저하가 아니라 **OOM**이다.

---

## 2. 문제 1 — 3계층 전제의 붕괴

Thor에서 `llama.cpp`는 다음을 보고한다.

```
Device 0: NVIDIA Thor, compute capability 11.0, VMM: yes, VRAM: 125748 MiB
```

**시스템 RAM 전체가 VRAM으로 보고된다.** 선행 시스템들이 설계 예산의 대부분을 쓰는 "GPU 메모리와
호스트 DRAM 사이에서 무엇을 언제 옮길 것인가"라는 질문이, 이 하드웨어에서는 **물리적 의미가 없다.**

| 시스템 | Venue | 전제 | Coherent SoC에서 |
|---|---|---|---|
| ZeRO-Infinity | SC'21 | GPU HBM → CPU DRAM → NVMe 분할 | 1·2계층 동일 → 분할이 **no-op** |
| FlashNeuron | **FAST'21** | GPUDirect로 GPU↔SSD 직결 | 별도 GPU 메모리 없음; Jetson의 cuFile은 compat 모드 정황 |
| DeepUM | ASPLOS'23 | UVM 페이지 마이그레이션 + 상관 프리페치 | **마이그레이션 대상지가 없음** |
| FlexGen | ICML'23 | 3단 GPU/CPU/디스크 블록 스케줄 탐색 | 두 단이 합쳐져 탐색공간 **퇴화** |
| G10 | MICRO'23 | 통합 GPU+호스트+플래시, 컴파일러 텐서 마이그레이션 | 가장 근접. 단 **tier1을 discrete GPU 메모리로 가정**하고 **시뮬레이션 평가** |
| PowerInfer | SOSP'24 | hot 뉴런 GPU 상주 / cold CPU | 같은 메모리 → 연산 배치만 바뀜 |
| InfiniGen | OSDI'24 | 투기적 KV 오프로드 GPU→CPU | 데이터 이동 no-op |
| NEO | MLSys'25 | attention/KV를 CPU로 오프로드 | 동일 |
| InstInfer / INF2 | '24–'25 | in-/near-storage 오프로드 | 직교. computational storage 필요 |
| **LLM in a Flash** | ACL'24 | 플래시→DRAM 윈도잉, 희소성 인지 로딩 | **최근접.** 그러나 CPU측에서 *무엇을* 로드할지 결정할 뿐, **GPU 가시 demand paging을 쓰지도 재지도 않으며** reclaim 실패를 다루지 않음 |

**방어할 경계.** 선행연구는 *물리적으로 분리된* GPU와 호스트 사이에서 **무엇을** 옮길지를 최적화한다.
여기선 둘이 분리돼 있지 않고, 남은 유일한 경계인 **GPU 가시 페이징 ↔ 플래시**의 **메커니즘**이 열린
문제다. 그리고 그 메커니즘은 방금 하드웨어에 생겼는데 — 작동하지 않는다.

---

## 3. 문제 2 — 하드웨어의 답이 장치의 96%를 버린다

Thor는 `pageableMemoryAccessUsesHostPageTables = 1` 인 **최초의 Tegra**다 (Orin은 0). 따라서 GPU
커널이 `mmap`된 파일 백업 메모리를 **직접 역참조**하고 OS가 폴트를 처리한다. 원리적으로는 애플리케이션
레벨 청킹 없이 out-of-core GPU 연산이 가능하다는 뜻이다.

실측하면 그 경로는 **장치 대역폭의 3~4%만 낸다.** 전체는 [`results/GRANULARITY.md`](results/GRANULARITY.md).

| 크기 | 경로 | 시간 | 실효 대역폭 | 장치(6.0 GB/s) 대비 |
|---:|---|---:|---:|---:|
| 32 GiB | OS demand paging (mmap + GPU fault) | 133.4 s | 0.240 GiB/s | 4.3% |
| 32 GiB | **gTier** (1 MiB 슬롯 × 8, io_uring) | **5.88 s** | **5.442 GiB/s** | **97.4%** |
| 64 GiB | OS demand paging | 343.0 s | 0.187 GiB/s | 3.3% |
| 64 GiB | **gTier** (256 MiB × 8) | **11.78 s** | **5.431 GiB/s** | **97.2%** |

**22.8x ~ 29.0x.**

### 왜인가 — 기법이 아니라 입도다

gTier의 슬롯 입도만 바꾸고 나머지를 전부 고정했다 (32 GiB, 슬롯 8개, 깊이 8):

| 슬롯 | 4 KiB | 16 KiB | 64 KiB | 256 KiB | **1 MiB** | 64 MiB | 256 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| GiB/s | **0.198** | 0.681 | 2.144 | 4.373 | **5.442** | 5.382 | 5.369 |

> **입도를 4 KiB로 낮추면 gTier가 OS 경로(0.240)와 똑같이 붕괴한다.**

이 NVMe 자체가 4 KiB에서 0.27~0.56 GiB/s, 64 KiB 이상에서 4.2~5.3 GiB/s를 낸다 — **10~20배 차이**다.
그리고 **GPU demand paging은 페이지 입도(4 KiB)에 구조적으로 고정**되어 있다.

**일반화되는 주장:** coherent SoC의 GPU demand paging은 OS나 드라이버를 어떻게 튜닝하든 현대 NVMe
대역폭의 **약 4%를 넘을 수 없다.** 해법은 **GPU의 접근 입도와 스토리지의 접근 입도를 분리**하는 것이다.

부수적으로, 64 KiB 이상에서는 **랜덤 ≈ 순차**다(4.311 vs 4.203 GiB/s). MoE의 전문가 단위 흩어진
접근은 그 자체로 문제가 아니며 스토리지 레이아웃 최적화는 이 플랫폼에서 불필요하다.

> ⚠️ **정정.** 이 저장소의 초기 커밋은 `ftruncate`로 만든 **희소 파일**로 측정해 "비결정적 하드 실패"를
> 보고했다. 희소 파일의 구멍을 읽으면 커널이 디스크를 건드리지 않고 제로 페이지를 할당하므로 그것은
> 스토리지가 아니라 페이지 할당을 잰 것이었다. **실데이터에서 mmap은 죽지 않는다** — 다만 위와 같이
> 치명적으로 느리다. 해당 주장은 철회하며 `results/REGIMES.md`에 정정 기록을 남겼다.

## 4. 접근 — gTier

OS의 demand paging 경로 대신, **GPU의 상주 윈도우를 유저스페이스에서 명시적으로 소유**하는 계층.
세 개의 하위 문제가 곧 연구 내용이다.

### ① 상주 관리 — 정확성
OS는 GPU 변환이 걸린 페이지를 모른 채 회수하고, 그래서 죽는다. gTier가 **경계 있는 상주 윈도우**를
직접 소유(`mlock` / `cudaHostRegister`)하여 커널이 GPU 밑에서 회수하지 못하게 하고, 축출을 명시적으로
수행한다.
> **연구 질문:** GPU를 멈추지 않으면서 윈도우 크기와 교체를 어떻게 결정하는가?

### ② 폴트 입도와 배칭 — 성능
4 KiB 폴트를 OS가 한 장씩 처리해 1.35 GiB/s에 머문다. GPU 폴트는 대규모 병렬로 도착하지만 서비스는
직렬이다. gTier는 **애플리케이션의 접근 구조를 알고**(MoE는 전문가 라우팅으로 흩어진 접근, dense는
레이어 순차) `io_uring`으로 큰 비동기 읽기를 GPU보다 앞서 발행한다.
> **연구 질문:** GPU 주도의 흩어진·위상 구조 폴트 스트림에 맞는 프리페치 정책은? 커널 readahead는
> CPU 순차 접근용이다.

### ③ 쓰기 반환과 대역폭 — 스토리지
더티 페이지를 소비자 플래시로 되돌려야 한다. 여기에 **(a)** CPU측 I/O가 GPU 메모리 대역폭을 훔치고
(측정: 0.912x), **(b)** 동시 읽기/쓰기가 NVMe 집계 처리량을 떨어뜨리며, **(c)** 소비자 SSD 수명이
걸린다.
> **연구 질문:** 공유 메모리 컨트롤러와 플래시 수명을 동시에 존중하는 writeback 스케줄링은?

---

## 5. 워크로드 설계

양자화를 **크기 손잡이로만** 사용한다. 모델 아키텍처·라우팅·접근 패턴을 고정한 채 오버서브스크립션
비율만 DRAM 경계를 가로질러 스윕한다.

### 주력 — Qwen3-235B-A22B-Instruct (MoE, 토큰당 22B 활성)
토큰당 가중치의 **약 9%만**, 전문가 라우팅에 따라 **흩어져서** 읽힌다. 커널의 순차 readahead에는 최악,
구조 인지 계층에는 최선인 케이스다.

| Quant | 크기 | 122.8 GiB 대비 |
|---|---:|---:|
| Q3_K_M | 104.7 GiB | 0.86x (들어감) |
| IQ4_XS | 116.9 GiB | 0.96x |
| Q4_0 | 124.0 GiB | 1.02x (막 넘음) |
| Q4_K_M | 132.4 GiB | 1.09x (초과) |
| Q5_K_M | 155.4 GiB | 1.27x |
| Q6_K | 179.8 GiB | 1.47x |
| Q8_0 | 232.8 GiB | 1.91x |

### 보조 — Qwen2.5 dense 사다리 (Q8_0)
7B (8 GiB) · 14B (16 GiB) · 32B (35 GiB) · 72B (77 GiB). §3에서 측정한 mmap regime
(fast / cliff / slow / slow)에 정확히 떨어지며, **dense vs MoE 접근 패턴 대조**를 제공한다.

---

## 6. 실험

**E1 — `-ngl` 스윕 (3계층 전제의 반증). ✅ 완료 — [결과](results/E1_NGL_SWEEP.md)**
discrete GPU에서 `-ngl K`는 몇 개 레이어가 VRAM에 상주하고 몇 개가 토큰마다 호스트에서 스트리밍되는지를
결정하며 처리량과 PCIe 트래픽을 지배한다. Thor에서 측정한 결과:

| 모델 | 처리량 변화 (`-ngl` 0→전체) | 디스크 읽기 변화 |
|---|---:|---:|
| Qwen2.5-7B Q8 | **2.94x (+194%)** | **+0.3%** |
| Qwen2.5-14B Q8 | **2.70x (+170%)** | **+0.2%** |

> 레이어를 CPU와 GPU 사이로 옮기면 처리량은 최대 2.94배 바뀌지만 **추가로 이동하는 바이트는
> 측정 오차 안에서 0이다.** `-ngl`은 이 하드웨어에서 데이터 배치 손잡이가 아니라 **연산 배치
> 손잡이**다 — 옮길 데이터가 없기 때문이다.

3계층 전제는 coherent 엣지 SoC에서 경험적으로 사망했다.

**E2 — DRAM 경계 통과.** MoE 양자화 사다리를 1.0x를 가로질러 스윕. 처리량, TTFT, major fault,
NVMe 읽기량, 페이지캐시 증가, 그리고 **완주 여부**를 기록.

**E3 — 메커니즘 비교.** `mmap`(OS demand paging) vs `--no-mmap`(명시적 read) vs gTier 프로토타입.
`--no-mmap`은 애초에 DRAM을 초과할 수 없다.

**E4 — MoE vs dense.** 같은 바이트, 다른 접근 구조. 전문가 단위로 흩어진 접근에서 커널 readahead는
도움이 되는가 해가 되는가?

**측정 지표** (전부 [`bench/bench.py`](bench/bench.py)가 포착): 처리량(pp/tg), wall time,
`pgmajfault`, `pgfault`, NVMe 섹터 읽기/쓰기, 페이지캐시 증가, MemAvailable, 보드 전력.

---

## 7. Go / Kill — **통과**

기준은 "경계 윈도우 프로토타입이 ① 죽지 않으면서 ② 1.35 GiB/s를 넘고, 이상적으로 장치의 6.0 GB/s에
근접하는가"였다.

**결과: 5.44 GiB/s = 장치의 97.4%, OS 경로 대비 22.8~29.0x.** 천장은 하드웨어가 아니었다.

### 반증된 가설 (정직한 음성 결과)

| 가설 | 결과 | 근거 |
|---|---|---|
| coherent SoC에서 CPU 작업은 공짜가 아니다 | **거짓** | CPU가 114.5 GiB/s를 써도 gTier 처리량 불변. out-of-core의 병목은 메모리가 아니라 스토리지이고 메모리 여유가 40배다 |
| MoE의 흩어진 접근에 레이아웃 최적화가 필요하다 | **거짓** | 64 KiB 이상에서 랜덤 ≈ 순차. 전문가 텐서는 이미 MiB 단위다 |
| GPU 페이징이 비결정적으로 붕괴한다 | **철회** | 희소 파일 아티팩트. 실데이터에서는 죽지 않는다 |

## 8. 저장소 구조

```
thor_gtier/
├── README.md               이 문서 — 노벨티와 주장
├── docs/
│   ├── PLATFORM_NOTES.md   Thor 실측 플랫폼 사실 (대역폭·용량·coherence·MIG·소프트웨어 공백)
│   └── EXPERIMENT_PLAN.md  실험 설계와 베이스라인 상세
├── mmap_probe/             GPU가 파일 백업 mmap을 역참조할 수 있는가 — §3의 세 regime 측정
│   ├── mmaptest.cu
│   └── Makefile
├── bw_probe/               CPU+GPU 동시 메모리 대역폭 — §1의 0.912x
│   ├── bw.cu
│   └── Makefile
├── bench/
│   └── bench.py            out-of-core LLM 추론 측정 하네스
├── scripts/
│   ├── fetch.sh            Qwen2.5 dense 사다리 다운로드
│   └── fetch2.sh           Qwen3-235B MoE 사다리 다운로드
└── results/
    └── REGIMES.md          mmap regime 원측정값
```

## 9. 재현

```bash
# 1) GPU mmap regime 특성화
cd mmap_probe && make
./mmaptest /path/to/file.bin 8      # 빠른 경로
./mmaptest /path/to/file.bin 24     # 절벽 이후
./mmaptest /path/to/file.bin 100    # 붕괴

# 2) 메모리 대역폭 포화
cd ../bw_probe && make && ./bw 12

# 3) 모델 준비 후 LLM 측정
cd .. && ./scripts/fetch.sh
python3 bench/bench.py \
  --models models/qwen7b_q8/*-00001-of-*.gguf \
  --ngl 0 20 40 99 --mmap 1 0 --pp 128 --tg 32 \
  --out results/e1_ngl_sweep.jsonl
```

측정 전 `sudo jetson_clocks`로 클럭을 고정하고, `nvpmodel -q`로 전력 모드를 기록할 것.
`bench.py`는 각 실행 전에 페이지 캐시를 비운다(`drop_caches`).
