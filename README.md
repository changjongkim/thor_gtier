# gTier — Coherent 엣지 SoC의 Out-of-Core GPU 데이터 경로

**플랫폼:** NVIDIA Jetson AGX Thor · JetPack 7.2 / L4T R39.2 · CUDA 13.0 · Blackwell sm_110
· 122.8 GiB 통합 coherent 메모리 · WD SN5000S 1 TB NVMe (PCIe Gen4 x4) · **swap 없음**

> **한 문장.** Coherent SoC에서는 GPU 메모리와 호스트 DRAM이 같은 물리 메모리라 기존 오프로딩
> 시스템이 최적화하는 계층이 사라진다. 남는 **DRAM↔플래시** 경계에서 성능을 결정하는 것은
> **읽는 바이트 수**와 **무엇을 상주시키느냐**이며, 전송 메커니즘은 장치 상한에 막혀 좁은 차이만
> 남긴다.

---

## 1. 서론

메모리보다 큰 모델을 돌리는 시스템 — ZeRO-Infinity, FlashNeuron, DeepUM, FlexGen, G10,
PowerInfer, InfiniGen, NEO — 은 모두 **GPU 메모리 / 호스트 DRAM / 스토리지** 세 계층을 전제하고,
설계 예산의 대부분을 1↔2계층 스케줄링에 쓴다.

Coherent 엣지 SoC에서는 그 전제가 성립하지 않는다. Thor에서 `llama.cpp`는 이렇게 보고한다.

```
Device 0: NVIDIA Thor, compute capability 11.0, VMM: yes, VRAM: 125748 MiB
```

**시스템 RAM 전체가 VRAM이다.**

그리고 Thor는 `pageableMemoryAccessUsesHostPageTables = 1` 인 최초의 Tegra다. GPU 커널이
`mmap`된 파일 백업 메모리를 직접 역참조하고 OS가 폴트를 처리한다 — 원리적으로 애플리케이션 레벨
청킹 없는 out-of-core GPU 연산이다. **그 경로는 장치 대역폭의 3~11%만 낸다.**

이 저장소는 그 원인을 분리하고, 무엇이 실제로 성능을 결정하는지 측정하며, 결과를 구현한 I/O 계층
`gTier`를 다섯 개 베이스라인과 **동일 인터페이스로** 비교한다.

### 측정된 결론 요약

| 레버 | 배율 | 상한 |
|---|---:|---|
| 전송 메커니즘 (gTier vs cuFile) | 1.12x (1 MiB) ~ 2.80x (16 KiB) | **SSD 5.682 GiB/s — 이미 97%** |
| 읽기 입도 | 10~20x | 장치 특성 |
| **라우팅 인지 (덜 읽기)** | **10.0x** | 활성 전문가 비율 |
| **상주 정책 (안 읽기)** | **최대 14x** | 메모리 대역폭 |
| 중복 제거 | **1.00x** | 모델 내 중복 0.02% |
| CPU 작업량 | **1.00x** | 스토리지 바운드 |
| GPU/IO 중첩 | **1.00x** | 교차점보다 세 자릿수 아래 |

---

## 2. 배경

### 2.1 플랫폼 실측

| 항목 | 측정값 |
|---|---:|
| GPU STREAM-triad | 250.2 GB/s (이론치 273의 91%) |
| CPU STREAM-triad (12스레드) | 209.2 GB/s |
| **CPU + GPU 동시** | **230.3 GB/s (0.912x)** |
| NVMe 링크 | PCIe Gen4 x4 = 이론 7.34 GiB/s |
| **NVMe 실측 최대** (fio 12설정) | **5.682 GiB/s** (1 MiB, 잡 1개, qd 64) |
| 최대 단일 매핑 할당 | 112 GiB |
| 핀드+매핑 할당 | 70 GiB 이상 확인 |
| Swap | **0 B** |

**GPU 혼자 메모리 컨트롤러를 포화시키므로** coherent CPU-GPU 협업은 메모리 바운드 작업에서
대역폭을 늘리지 못한다. **폴백도 없다** — 작업집합이 DRAM을 넘으면 OOM이다.

**병목은 링크가 아니라 SSD다.** Gen4 x4가 7.34 GiB/s를 나를 수 있는데 이 DRAM-less 클라이언트
드라이브가 5.68에서 막는다. 그리고 **동시성을 늘리면 오히려 느려진다** — 잡 1개 5.682,
4개 5.102. 단일 스트림 qd 64에서 이미 포화한다.

### 2.2 Coherent SoC가 바꾸는 것

| | discrete GPU | **Thor** |
|---|---|---|
| GPU 메모리 | 별도 HBM/GDDR | **시스템 RAM과 동일** |
| 호스트→디바이스 복사 | PCIe 전송 | DRAM→DRAM (127 GB/s) |
| GPUDirect Storage | NVMe → VRAM 직접 | **없음** (`nvfs 0.0`, compat 모드) |
| GPU가 파일 백업 mmap 접근 | 불가 | **가능** |

### 2.3 NVIDIA GDS는 이 플랫폼에 없다

`cuFileDriverGetProperties` 직접 조회:

```
cuFile driver: nvfs major=0 minor=0     ← nvidia-fs 커널 드라이버 없음
dstatusflags=0x0  →  GDS supported = 0  ← POSIX compat 모드 (바운스 버퍼)
```

Thor에서 cuFile은 `NVMe → 바운스 → cudaMemcpy → cudaMalloc 버퍼` 이며 복사 제거 이점이 없다.
다만 **버퍼 등록 자체에는 한계가 없다** — 64 MiB 버퍼 640개, **40 GiB 등록이 정상 동작한다**
([`gtier/cufile_limit.cu`](gtier/cufile_limit.cu)).

---

## 3. 설계

### 3.1 데이터 경로

```
discrete + GDS   NVMe ──DMA──▶ VRAM ──▶ GPU                       복사 0
Thor + cuFile    NVMe ─▶ 바운스 ─▶ cudaMemcpy ─▶ 버퍼 ─▶ GPU      복사 2
Thor + mmap      NVMe ─▶ 페이지캐시 ─▶ GPU 폴트(4 KiB)            복사 1, 25배 느림
gTier            NVMe ──DMA──▶ cudaHostAllocMapped ──▶ GPU        복사 0
```

`cudaHostAlloc(cudaHostAllocMapped)` 메모리는 동시에 (a) O_DIRECT DMA 대상이고 (b) GPU가 직접
주소 지정하는 메모리다. **드라이브가 GPU가 읽을 메모리에 직접 쓴다.**

### 3.2 API — 엔진이 아니라 I/O 계층

```c
gtier *g = gtier_open(path, &cfg);
gtier_fetch(g, ranges, n, dev_out);            // 동기
gtier_submit(g, ranges, n, &ticket);           // 비동기: 큐를 비우지 않는다
gtier_wait(g, &ticket, dev_out);
```

클라이언트는 out-of-core GPU 워크로드 일반 — LLM 가중치, 양자 상태벡터, 벡터 인덱스, DB 버퍼풀.

### 3.3 Read planner — 두 규칙 모두 측정에서 도출

**규칙 1 — 고립된 범위를 증폭하지 마라.** 장치의 대역폭-크기 곡선이 선형보다 느리게 자라므로
증폭이 언제나 이득을 앞지른다(§4.4).

**규칙 2 — 인접 범위도 병합하지 마라.** 깊은 큐에서 요청은 거의 공짜이고 바이트만 값을 한다(§4.5).

**보정은 모델이 아니라 측정으로.** 파라메트릭 모델(고정 요청비용 + 바이트당 비용)로 임계값을
유도했다가 **틀렸다** — 이 장치의 곡선은 선형도 단조도 아니다(랜덤 256 KiB 1.179 GiB/s가
64 KiB의 4.311보다 느리다). 현재는 planner를 후보 임계값들로 실제 돌려 최선을 고른다.

### 3.4 연속 제출

`gtier_fetch`는 제출 후 전부 기다려 **인출 사이에 큐가 0으로 비워진다.** 티켓 기반 submit/wait이
다음 배치를 먼저 내보내고 이번 것을 거두어 링을 채운 상태로 유지한다. 티켓은 겹치지 않는 슬롯을
소유하고, CQE의 `user_data`에 티켓 id를 실어 **다른 티켓의 완료를 만나도 잃지 않는다**(§4.6).

### 3.5 상주 정책

캐시는 고정 블록을 요구하고 그것은 규칙 1이 금지하는 증폭이다. 재사용이 있으면 상환되므로
교차점이 존재하고, 그 교차점은 **240배로 날카롭다**(§4.8).

| 정책 | 동작 |
|---|---|
| `NONE` | 모든 범위를 정확히 인출 |
| `BLOCK` | 고정 블록 + LRU |
| `HYBRID` | 미스는 정확 인출, N회 접근 시 승격 — **실패**(§4.8) |
| `ADAPTIVE` | 히트율 관측 후 체제 전환, 히스테리시스 40%/70% |
| `PIN` | 윈도우를 한 번 채우고 **축출 중단** — 순환 스캔용(§4.9) |

---

## 4. 평가

모든 측정: 실데이터(비희소·비압축) 파일, 실행마다 페이지 캐시 비움, `jetson_clocks` 고정,
백엔드마다 별도 프로세스.

### 4.1 3계층 전제의 반증

| 모델 | 처리량 변화 (`-ngl` 0→전체) | 디스크 읽기 변화 |
|---|---:|---:|
| Qwen2.5-7B Q8 (7.5 GiB) | **2.94x** | **+0.3%** |
| Qwen2.5-14B Q8 (14.6 GiB) | **2.70x** | **+0.2%** |
| Qwen2.5-32B Q8 (32.4 GiB) | **2.06x** | **+0.14%** |

> 처리량은 최대 2.94배 바뀌지만 **추가로 이동하는 바이트는 측정 오차 안에서 0이다.**
> `-ngl`은 데이터 배치가 아니라 **연산 배치** 손잡이다.

[`results/E1_NGL_SWEEP.md`](results/E1_NGL_SWEEP.md)

### 4.2 원인 — readahead가 GPU 폴트에 닿지 않는다

같은 파일, 같은 매핑, 같은 접근 패턴. **누가 폴트하느냐만** 다르다.

| 폴트 주체 | 처리량 |
|---|---:|
| **GPU 전체** | **0.220 GiB/s** |
| CPU 스레드 **1개** | **2.736 GiB/s** |

**CPU 스레드 하나가 GPU 전체보다 12.4배 빠르다.** mmap도, 4 KiB 페이지 자체도, 스토리지도
원인이 아니다.

### 4.3 메커니즘은 입도다

gTier의 슬롯 입도만 바꾸고 나머지를 고정:

| 슬롯 | 4 KiB | 16 KiB | 64 KiB | 256 KiB | **1 MiB** | 256 MiB |
|---|---:|---:|---:|---:|---:|---:|
| GiB/s | **0.198** | 0.681 | 2.144 | 4.373 | **5.442** | 5.369 |

**4 KiB를 주면 gTier도 OS 경로(0.240)와 똑같이 붕괴한다.**

[`results/GRANULARITY.md`](results/GRANULARITY.md)

### 4.4 입도 바닥 — 증폭은 손해다

유효 대역폭 = (항목 수 × g) / 경과시간.

| g | 최적 블록 | 최적 유효 대역폭 | 1 MiB 대비 |
|---:|---:|---:|---:|
| 4 KiB | 4 KiB | 0.530 GiB/s | **0.12x** |
| 16 KiB | 32 KiB | 1.410 | 0.33x |
| **64 KiB** | 64 KiB | 3.146 | 0.73x |
| 1 MiB | 1 MiB | **4.324** | 1.00x |

g=4 KiB에서 64 KiB 블록으로 키우면 raw가 8.2배 오르지만 증폭 16배라 유효 대역폭은 **0.51배로
악화**한다. *"큰 블록으로 읽어 대역폭을 얻어라"* 는 순차 접근에서만 맞다.

**함의:** miss가 플래시로 갈 때 희소성 활용 입도에 상한이 생긴다. 전문가(수십~수백 MiB)와
레이어(수백 MiB)는 위에 있고, **PowerInfer의 뉴런(수~수십 KiB)과 KV 페이지는 아래**라 3~8배를 잃는다.

[`results/GRANULARITY_FLOOR.md`](results/GRANULARITY_FLOOR.md)

### 4.5 병합하지 마라

인접 범위, 64 KiB × 32:

| 임계값 | 요청/인출 | 증폭 | 유효 대역폭 |
|---|---:|---:|---:|
| **병합 없음** | 32.0 | 1.00x | **2.054 GiB/s** |
| 64 KiB | 4.0 | 1.88x | 1.908 (−7%) |
| 1 MiB | 4.0 | 1.88x | 1.752 (−15%) |

얕은 큐(n=4)에서도 1.046 → 0.782 (−25%).

### 4.6 연속 제출

1 MiB 항목:

| 배치 n | 동기 | **비동기** | 개선 |
|---:|---:|---:|---:|
| 8 | 3.831 | **5.044** | **+31.7%** |
| 16 | 4.210 | 5.126 | +21.8% |
| 32 | 4.517 | 5.166 | +14.4% |
| 64 | 4.738 | 5.203 | +9.8% |

[`results/ASYNC_FIX.md`](results/ASYNC_FIX.md)

### 4.7 백엔드 비교

**측정의 의미.** 모든 숫자는 **유효 대역폭(GiB/s)** — 호출자가 요청한 바이트 ÷ 경과 시간.
아래는 모두 증폭 1.00배다. `item` = 요청 한 건의 크기, `n` = 인출당 범위 개수.
**조건:** 32 GiB 실데이터, 16 GiB 구간 무작위 오프셋, 캐시 비움.

| 백엔드 | 하는 일 | 대응하는 선행 설계 |
|---|---|---|
| **gtier** | NVMe가 GPU 주소지정 메모리에 직접 DMA | 본 연구 |
| cufile | NVIDIA GDS API (여기선 compat 모드) | NVIDIA 공식 |
| pread+copy | `pread` → `cudaMemcpy` → 디바이스 버퍼 | FlexGen / ZeRO-Infinity |
| mmap-cpu | mmap 후 CPU 스레드가 폴트 | 커널 readahead 경로 |
| uvm | `cudaMallocManaged` + prefetch | DeepUM |
| mmap-gpu | mmap 후 **GPU가 폴트** | Thor 하드웨어 경로 |

| item | n | **gtier** | cufile | pread+copy | mmap-cpu | uvm | mmap-gpu | vs cuFile |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 KiB | 256 | **1.817** | 0.650 | 0.658 | 0.114 | 0.125 | 0.040 | **2.80x** |
| 64 KiB | 64 | **3.281** | 1.720 | 1.609 | 0.310 | 0.356 | 0.120 | **1.91x** |
| 256 KiB | 16 | **4.059** | 2.782 | 2.645 | 0.847 | 0.812 | 0.299 | **1.46x** |
| 1 MiB | 8 | **5.084** | 3.960 | 3.695 | 1.368 | 1.327 | 0.727 | **1.28x** |
| 4 MiB | 8 | **5.154** | 4.459 | 4.251 | 3.218 | 1.729 | 1.536 | **1.16x** |

**작은 입도일수록 유리한 이유는 복사가 대역폭이 아니라 전송당 고정비이기 때문이다.**
`cudaMemcpy` H2D는 4 KiB~256 KiB에서 전부 19~21 µs로 같고(런치·동기화 비용), 1 MiB 위에서야
바이트당 비용이 127~142 GB/s로 드러난다. 읽기가 20 µs보다 짧을 때 — **128 KiB 아래** — 만
복사가 지배한다. 같은 하네스에서 gTier에 복사를 인위적으로 붙여 격리하면 16 KiB 26.1%,
64 KiB 17.4%, 1 MiB 4.7%다.

[`results/BACKENDS.md`](results/BACKENDS.md) · [`results/COPY_COST.md`](results/COPY_COST.md)

#### CPU와 메모리 비용

1 MiB × 64, 32회:

| 백엔드 | GiB/s | CPU 코어 | 프로세스 RSS |
|---|---:|---:|---:|
| **gtier** | **5.184** | **1.2** | **0.32 GiB** |
| cufile | 4.857 | 3.0 | 1.08 GiB |
| pread+copy | 4.602 | 4.3 | 0.13 GiB |

**gTier가 더 빠르면서 CPU를 2.5배 적게 쓴다.** 그리고 이 장치는 동시성을 늘리면 느려지므로
(잡 1개 5.682 vs 4개 5.102) 코어 절약이 곧 옳은 설계다.

### 4.8 상주 — 체제 감지기가 블록 휴리스틱을 이긴다

64 KiB 항목, 1 MiB 블록(증폭 16x), 윈도우 64 블록:

| 워킹셋 | exact | block | HYBRID | **ADAPTIVE** |
|---:|---:|---:|---:|---:|
| 8 | 3.749 | 78.144 | 80.62 | **82.395** |
| 32 | 2.858 | 80.411 | 78.57 | **81.484** |
| 64 | 2.708 | **81.221** | 25.85 | 80.883 |
| 96 | **2.712** | 0.331 | 1.09 | 2.306 |
| 512 | **2.717** | 0.335 | 0.65 | 2.323 |

**순수 정책은 각각 한쪽 끝에서 7~8배 진다.** HYBRID는 실패했다 — "N회 접근 시 승격"이
*곧 재사용* 과 *언젠가 재사용* 을 구분하지 못한다. ADAPTIVE는 전 구간에서 최선의 15% 이내다.

> **히트 시 79~81 GiB/s는 장치 최대(5.68)의 14배다.** 스토리지를 아예 안 가기 때문이다.
> 유효 대역폭이 장치 상한을 넘을 수 있는 유일한 경우다.

### 4.9 순환 스캔에서 LRU는 병리적이다

가중치 스트리밍은 토큰마다 모델 전체를 한 번 순환 스캔한다. LRU의 최악 케이스다.

**Qwen2.5-32B (32.42 GiB), 토큰 3회:**

| 윈도우 | BLOCK(LRU) tok/s | **PIN tok/s** | PIN 이득 |
|---:|---:|---:|---:|
| 13.9% | 0.1570 | 0.1626 | +4% |
| 27.8% | 0.1601 | **0.1779** | +11% |
| 55.5% | 0.1537 | **0.2190** | **+42%** |
| 83.3% | 0.1642 | **0.3046** | **+85%** |
| 104.1% | **0.4405** | 0.3999 | −9% |

**Qwen3-235B-A22B Q4_K_M (132.4 GiB > DRAM 122.8 GiB), 토큰 3회:**

| 윈도우 | 모델 대비 | LRU tok/s | **PIN tok/s** | **PIN 이득** | PIN 히트 |
|---:|---:|---:|---:|---:|---:|
| 12 GiB | 9.1% | 0.0388 | 0.0422 | +9% | 7.1% |
| 24 GiB | 18.1% | 0.0380 | 0.0454 | +19% | 14.2% |
| 48 GiB | 36.3% | 0.0380 | 0.0497 | +31% | 28.5% |
| 72 GiB | 54.4% | 0.0379 | 0.0572 | **+51%** | 42.8% |
| **96 GiB** | **72.5%** | 0.0378 | **0.0689** | **+82%** | **57.1%** |

**LRU는 윈도우를 8배 키워도 0.0378~0.0388로 완전히 평평하다.** PIN은 히트율이 윈도우에
비례해 올라간다.

[`results/RESIDENCY_REAL.md`](results/RESIDENCY_REAL.md)

### 4.10 실제 모델 트레이스

**Dense — Qwen2.5-32B Q8_0 (32.42 GiB, DRAM에 들어감), 콜드 캐시:**

| 백엔드 | GiB/s | tok/s |
|---|---:|---:|
| **gtier (비동기)** | **5.411** | **0.1669** |
| gtier (동기) | 4.911 | 0.1515 |
| cufile | 4.324 | 0.1333 |
| pread+copy | 3.992 | 0.1231 |
| mmap-cpu | 3.784 | 0.1167 |
| uvm | 2.716 | 0.0838 |
| mmap-gpu | 1.445 | 0.0446 |

**DRAM 초과 — Qwen3-235B-A22B Q4_K_M (132.4 GiB vs 122.8 GiB):**

| 백엔드 | GiB/s | tok/s |
|---|---:|---:|
| **gtier (비동기)** | **5.516** | **0.0417** |
| gtier (동기) | 5.182 | 0.0391 |
| cufile | 4.917 | 0.0371 |
| pread+copy | 4.450 | 0.0336 |
| uvm | 2.642 | 0.0200 |
| **mmap-gpu** | **0.641** | 0.0048 |
| **mmap-cpu** | **0.567** | 0.0043 |

> **모델이 DRAM을 넘자 mmap 경로 둘 다 붕괴한다** — dense에서 1.445/3.784였던 것이
> 0.641/0.567로. 페이지 캐시가 더는 담지 못해 순수 폴트 비용만 남는다. **gTier는 변하지 않는다.**

### 4.11 MoE 라우팅 — 가장 큰 레버

전문가 텐서가 모델의 **96.6%**(127.86 GiB)를 차지하고, 레이어당 128개가 쌓인 형태다.
Qwen3-235B-A22B는 토큰당 **8/128 = 6.25%** 만 활성이다.

| 트레이스 | GiB/s | **tok/s** | 토큰당 |
|---|---:|---:|---:|
| dense 전체 스윕 | 5.421 | 0.0409 | 132.4 GiB |
| **라우팅 8/128** | 5.125 | **0.4095** | ~12.5 GiB |
| 라우팅 16/128 | 5.232 | 0.2552 | ~20 GiB |
| 라우팅 32/128 | 5.298 | 0.1452 | ~36 GiB |

**10배 개선이고 전적으로 덜 읽어서 얻은 것이다** — 대역폭은 그대로다.

같은 라우팅(8/128)에서 백엔드 비교: gtier **0.4129** > cufile 0.3776 > pread 0.3460 >
mmap-gpu 0.0946 tok/s.

### 4.12 값을 하지 않는 것들 (음성 결과)

| 가설 | 결과 | 근거 |
|---|---|---|
| coherent SoC에서 CPU 작업은 공짜가 아니다 | **거짓** | 배경 CPU 부하 114.5 GiB/s에도 gTier 불변 |
| GPU/IO 중첩이 값을 한다 | **거짓** | 전 백엔드 0.95~1.05x. 교차점 ~1000 FLOPs/byte인데 LLM 디코드는 2~4, 상태벡터는 0.38 |
| MoE 흩어진 접근에 레이아웃 최적화가 필요하다 | **거짓** | 64 KiB 이상에서 랜덤 ≈ 순차 |
| 모델에 중복 콘텐츠가 있다 | **거짓** | 블록 중복률 **0.00~0.02%** |
| io_uring 미세 최적화가 도움된다 | **거짓** | regbuf +0.15%, SQPOLL 불변, **IOPOLL −29%** |

[`results/PIPELINE.md`](results/PIPELINE.md)

---

## 5. 정직한 한계와 철회

- **전송 메커니즘의 우위는 큰 입도에서 좁다.** 1 MiB에서 cuFile 대비 1.28배, 4 MiB에서 1.16배다.
  LLM 가중치는 MiB 단위를 읽으므로 실제 운용 구간에서 격차가 작다. **레버는 전송이 아니라
  상주(최대 14x)와 덜 읽기(10x)에 있다.**
- **장치가 상한이다.** SSD 실측 최대 5.682 GiB/s(PCIe는 7.34까지 가능)이고 gTier는 97%에 있다.
  남은 3%는 비동기 경로가 티켓 2개만 써서 GPU 커널이 완전히 가려지지 않는 우리 쪽 결함이다.
- **철회: "cuFile이 큰 윈도우를 못 잡는다".** 직접 시험하니 **64 MiB 버퍼 640개, 40 GiB 등록이
  정상 동작한다.** 앞선 8 GiB 실패는 우리 벤치마크 설정 탓이었다.
- **철회: "비결정적 붕괴".** `ftruncate`가 만든 희소 파일 아티팩트였다
  ([`results/REGIMES.md`](results/REGIMES.md)).
- **정정: zero-copy는 대역폭이 아니라 지연 이득이다.** 복사는 전송당 고정비 ~20 µs이고
  coherent SoC에서 복사 대역폭(127 GB/s)은 스토리지보다 23배 빠르다.
- **탑티어 베이스라인을 실제로 돌리지 못했다.** MoE-Infinity, PowerInfer, FlexGen 비교에는
  sm_110용 PyTorch가 필요한데 PyPI aarch64 휠은 CPU 전용이다(`2.9.1+cpu`).
- **MoE 라우팅이 실행당 한 번만 뽑힌다.** 모든 토큰이 같은 전문가를 읽으므로 캐시에 최선인
  경우다. 토큰별 재라우팅 하에서 상주 정책이 얼마나 버티는지는 미측정이다.
- **`gtier_fetch`의 동기 경로는 여전히 큐를 비운다.** 비동기 API가 있지만 캐시 정책과는 아직
  결합되지 않았다(`gtier_submit`은 `GTIER_CACHE_NONE`만 지원).

---

## 6. 저장소 구조

```
thor_gtier/
├── README.md                    이 문서
├── docs/
│   ├── PLATFORM_NOTES.md        Thor 실측 플랫폼 사실
│   └── EXPERIMENT_PLAN.md       실험 설계와 베이스라인 상세
├── lib/                         gTier 라이브러리 + 다섯 베이스라인
│   ├── gtier.h / gtier.cu       범위 인출 API, planner, 상주 정책, 백엔드
│   ├── gguf.h / gguf.c          GGUF 파서
│   ├── bench.cu                 합성 워크로드 드라이버
│   ├── gguf_bench.cu            실제 가중치 트레이스 (MoE 라우팅 포함)
│   ├── run_bench.sh             백엔드별 캐시 비움 드라이버
│   └── run_pipe.sh              파이프라이닝 비교
├── gtier/                       프로브
│   ├── gtier.cu                 OS 경로 vs 윈도우
│   ├── randread.c               NVMe 입도별 랜덤·순차
│   ├── sparsefetch.c            입도 바닥 전면 측정
│   ├── cufile_test.cu           GDS 가용성 검증
│   ├── cufile_limit.cu          cuFile 버퍼 등록 한계
│   └── dedup.c                  모델 내 중복 블록 측정
├── mmap_probe/ · bw_probe/      GPU mmap regime · CPU+GPU 대역폭
├── bench/                       llama.cpp 측정 하네스
├── scripts/
│   ├── fetch.sh / fetch2.sh     모델 다운로드
│   ├── run_queue.sh             무인 실험 큐 (끊겨도 계속, 단계별 커밋)
│   └── summarize.py             큐 결과 요약 생성
└── results/                     모든 측정 결과
    ├── GRANULARITY.md           §4.2~4.3
    ├── GRANULARITY_FLOOR.md     §4.4
    ├── BACKENDS.md              §4.5, §4.7, §4.8
    ├── ASYNC_FIX.md             §4.6
    ├── COPY_COST.md             §4.7 복사 비용 격리
    ├── RESIDENCY_REAL.md        §4.9
    ├── PIPELINE.md              §4.12
    ├── E1_NGL_SWEEP.md          §4.1
    ├── E2_PARTIAL.md            cgroup 에뮬레이션의 방법론적 한계
    ├── LIBRARY.md               planner 설계 기록
    ├── REGIMES.md               철회된 초기 측정
    └── auto/SUMMARY.md          무인 큐 자동 요약
```

---

## 7. 설치 및 사용

### 7.1 요구사항

- NVIDIA Jetson AGX Thor (JetPack 7.2 / L4T R39.2 이상), CUDA 13.0, `sm_110`
- `liburing-dev` (2.5 이상), `gcc`, `nvcc`
- O_DIRECT를 지원하는 파일시스템 위의 NVMe
- 페이지 캐시를 비우려면 passwordless `sudo` (측정 재현용)

```bash
sudo apt install -y liburing-dev
```

### 7.2 빌드

```bash
git clone git@github.com:changjongkim/thor_gtier.git
cd thor_gtier/lib
make                      # libgtier.a, gtier_bench, gguf_bench
make ARCH=sm_87           # 다른 Jetson 세대
```

### 7.3 라이브러리 사용

```c
#include "gtier.h"

gtier_config cfg = {0};
cfg.backend      = GTIER_BACKEND_GTIER;
cfg.slot_bytes   = 1 << 20;            // 측정된 최적 스테이징 크기
cfg.slots        = 64;
cfg.queue_depth  = 64;
cfg.merge_gap    = 0;                  // 측정 결과: 병합하지 않음
cfg.cache_policy = GTIER_CACHE_ADAPTIVE;
cfg.max_fetch_ranges = 32;

gtier *g = gtier_open("model.gguf", &cfg);

gtier_range r[32] = { {off0, len0}, {off1, len1} /* ... */ };
void *dev[32];
gtier_fetch(g, r, 32, dev);            // dev[i] 는 GPU가 바로 읽는 포인터
my_kernel<<<...>>>((const uint8_t *)dev[0], len0);
gtier_close(g);
```

**연속 배치를 내는 클라이언트는 비동기 API를 쓴다** — 큐를 비우지 않아 얕은 깊이에서 최대 31.7% 빠르다.

```c
gtier_ticket t0, t1;
gtier_submit(g, r0, n, &t0);
for (;;) {
    gtier_submit(g, r1, n, &t1);       // 다음 배치를 먼저
    gtier_wait(g, &t0, dev);           // 그다음 이번 것을 거둔다
    consume(dev);
    std::swap(t0, t1);
}
```

**제약.** 한 범위는 하나의 슬롯 안에 들어가야 한다. O_DIRECT 정렬이 양끝으로 최대 4 KiB씩
늘어나므로 범위 길이를 `slot_bytes - 8192` 이하로 유지한다. 캐시 정책을 쓸 때는 범위가 블록
경계를 걸치지 않아야 한다. 위반 시 `-E2BIG`. 비동기 API는 현재 `GTIER_CACHE_NONE`만 지원한다.

```bash
nvcc -O3 -std=c++17 -arch=sm_110 myapp.cu libgtier.a -luring -lcufile \
  -I/usr/local/cuda/targets/sbsa-linux/include \
  -L/usr/local/cuda/targets/sbsa-linux/lib -o myapp
```

### 7.4 측정 재현

```bash
sudo jetson_clocks
nvpmodel -q
```

**주의:** 테스트 파일은 반드시 **실제 데이터**여야 한다. `ftruncate`로 만든 희소 파일은 구멍을
읽을 때 커널이 디스크를 건드리지 않고 제로 페이지를 할당하므로, 스토리지가 아니라 페이지 할당을
측정하게 된다.

```bash
openssl enc -aes-256-ctr -pass pass:gtier -nosalt < /dev/zero 2>/dev/null \
  | dd of=/tmp/real32.bin bs=64M count=512 iflag=fullblock oflag=direct
du -h --apparent-size /tmp/real32.bin && du -h /tmp/real32.bin   # 둘이 같아야 함
```

```bash
cd lib
./run_bench.sh --file /tmp/real32.bin --item 65536 --n 64 --iters 64     # 백엔드 비교
./gtier_bench --file /tmp/real32.bin --item 65536 --n 64 --slots 64 \
  --policy 3 --reuse 128 --iters 256 --only 0                            # 상주 정책
./run_pipe.sh --file /tmp/real32.bin --item 1048576 --n 16 --iters 64 --policy 0
```

실제 모델:

```bash
./gguf_bench $(for f in /path/to/model/*.gguf; do echo --shard $f; done) \
  --backend 0 --policy 0 --tokens 1 --slot 1048576 --slots 576 --async 1

# MoE 라우팅 반영
./gguf_bench $(...) --backend 0 --policy 0 --tokens 1 --slot 1048576 \
  --slots 576 --async 1 --experts 128 --active 8 --skew 1.0
```

`--backend` 0=gtier, 1=mmap-gpu, 2=mmap-cpu, 3=pread+copy, 4=cufile, 5=uvm.
`--policy` 0=NONE, 1=BLOCK, 2=HYBRID, 3=ADAPTIVE, 4=PIN.

### 7.5 무인 실험 큐

연결이 끊겨도 계속 돌고 단계마다 커밋·푸시한다. 재실행하면 완료된 단계는 건너뛴다.

```bash
setsid nohup ./scripts/run_queue.sh > results/auto/queue.out 2>&1 < /dev/null &
tail -f results/auto/progress.log
```

결과 요약은 [`results/auto/SUMMARY.md`](results/auto/SUMMARY.md)에 자동 생성된다.

```bash
./scripts/fetch.sh      # Qwen2.5 dense 사다리
./scripts/fetch2.sh     # Qwen3-235B-A22B MoE
```
