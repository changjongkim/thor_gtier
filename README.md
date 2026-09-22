# gTier — Coherent 엣지 SoC의 Out-of-Core GPU 데이터 경로

**플랫폼:** NVIDIA Jetson AGX Thor · JetPack 7.2 / L4T R39.2 · CUDA 13.0 · Blackwell sm_110
· 122.8 GiB 통합 coherent 메모리 · WD SN5000S 1 TB NVMe · **swap 없음**

> **한 문장.** Coherent SoC에서는 GPU 메모리와 호스트 DRAM이 같은 물리 메모리라 기존
> 오프로딩 시스템이 최적화하는 계층이 사라지고, 남는 **DRAM↔플래시** 경계에서는
> **읽는 바이트 수**와 **무엇을 상주시키느냐** 두 가지만이 성능을 결정한다.
> 어떻게 옮기는지, 언제 겹치는지, CPU를 얼마나 쓰는지는 값을 하지 않는다.

---

## 1. 서론

메모리보다 큰 모델을 돌리는 시스템 — ZeRO-Infinity, FlashNeuron, DeepUM, FlexGen, G10,
PowerInfer, InfiniGen, NEO — 은 모두 **세 개의 구분된 계층**을 전제한다: GPU 메모리,
호스트 DRAM, 스토리지. 설계 예산의 대부분이 1계층과 2계층 사이에서 무엇을 언제 옮길지를
스케줄링하는 데 쓰인다.

Coherent 엣지 SoC에서는 그 전제가 성립하지 않는다. Thor에서 `llama.cpp`는 이렇게 보고한다.

```
Device 0: NVIDIA Thor, compute capability 11.0, VMM: yes, VRAM: 125748 MiB
```

**시스템 RAM 전체가 VRAM이다.** 1·2계층 사이의 이동은 물리적으로 존재하지 않는다.

그리고 Thor는 `pageableMemoryAccessUsesHostPageTables = 1` 인 최초의 Tegra다. GPU 커널이
`mmap`된 파일 백업 메모리를 **직접 역참조**하고 OS가 폴트를 처리한다 — 원리적으로 애플리케이션
레벨 청킹 없는 out-of-core GPU 연산이다. 실측하면 그 경로는 **장치 대역폭의 3~4%**만 낸다.

이 저장소는 (a) 왜 그런지 원인을 분리하고, (b) 무엇이 실제로 성능을 결정하는지 측정하며,
(c) 그 결과를 구현한 I/O 계층 `gTier`와 다섯 개 베이스라인을 동일 인터페이스로 비교한다.

### 기여

1. **새 하드웨어 경로의 첫 특성화.** GPU 폴트 경로는 장치의 3.9%를 낸다. 같은 매핑·같은 접근
   패턴에서 **CPU 스레드 하나가 GPU 전체보다 12.4배 빠르다** — 커널 readahead가 GPU 폴트에
   닿지 않기 때문이다. 원인이 입도임을 한 시스템에서 양 끝점을 재현해 확정했다(§4.2).
2. **입도 바닥.** 희소 접근에서 **읽기 증폭은 거의 언제나 손해**이며, 유효 대역폭은 희소성
   입도만의 함수가 되고 **64 KiB 아래에서 무너진다.** 이는 miss가 플래시로 갈 때 희소성을
   얼마나 잘게 활용할 수 있는지에 정량적 상한을 준다(§4.3).
3. **병합하지 마라.** 스토리지 시스템의 보편적 조언인 I/O 병합이 **모든 큐 깊이에서 손해**다(§4.4).
4. **상주는 체제 문제다.** 캐싱 여부가 **240배**를 가르고 두 순수 정책은 반대 방향으로 실패한다.
   블록 단위 승격 휴리스틱은 실패하고, **체제 감지기**가 전 구간에서 최선의 15% 이내에 머문다(§4.5).
5. **세 개의 음성 결과가 설계 공간을 붕괴시킨다.** CPU 작업, GPU/IO 중첩, 전송 메커니즘은
   이 플랫폼에서 값을 하지 않는다(§4.6).

---

## 2. 배경

### 2.1 실측한 플랫폼 사실

전체는 [`docs/PLATFORM_NOTES.md`](docs/PLATFORM_NOTES.md).

| 항목 | 측정값 |
|---|---:|
| GPU STREAM-triad | 250.2 GB/s (이론치 273의 91%) |
| CPU STREAM-triad (12스레드) | 209.2 GB/s |
| **CPU + GPU 동시** | **230.3 GB/s (0.912x)** |
| NVMe 순차 읽기 (O_DIRECT, 실데이터) | 6.0 GB/s = 5.59 GiB/s |
| **DRAM : NVMe** | **약 45 : 1** |
| 최대 단일 매핑 할당 | 112 GiB |
| Swap | **0 B** |

두 가지가 따라 나온다. **GPU 혼자 메모리 컨트롤러를 포화시키므로** coherent CPU-GPU 협업은
메모리 바운드 작업에서 대역폭을 늘리지 못한다. 그리고 **폴백이 없다** — 작업집합이 DRAM을
넘으면 우아한 저하가 아니라 OOM이다.

### 2.2 Coherent SoC가 바꾸는 것

| | discrete GPU | **Thor** |
|---|---|---|
| GPU 메모리 | 별도 HBM/GDDR | **시스템 RAM과 동일** |
| 호스트→디바이스 복사 | PCIe 전송 | **존재하지 않음** |
| GPUDirect Storage | NVMe → VRAM 직접 | **없음** (검증: `nvfs 0.0`) |
| GPU가 파일 백업 mmap 접근 | 불가 | **가능** (`pageableMemoryAccessUsesHostPageTables=1`) |

### 2.3 NVIDIA의 GDS는 이 플랫폼에 없다

`cuFileDriverGetProperties`를 직접 조회했다([`gtier/cufile_test.cu`](gtier/cufile_test.cu)).

```
cuFile driver: nvfs major=0 minor=0     ← nvidia-fs 커널 드라이버 없음
dstatusflags=0x0  →  GDS supported = 0  ← POSIX compat 모드 (바운스 버퍼)
```

`libcufile.so 1.15.0`은 설치되어 있지만, Thor에서 cuFile은
`NVMe → 바운스 → cudaMemcpy → cudaMalloc 버퍼` 이며 GPUDirect의 복사 제거 이점이 없다.

---

## 3. 설계

### 3.1 데이터 경로

```
discrete + GDS   NVMe ──DMA──▶ VRAM ──▶ GPU                       복사 0
Thor + cuFile    NVMe ─▶ 바운스 ─▶ cudaMemcpy ─▶ 버퍼 ─▶ GPU      복사 2
Thor + mmap      NVMe ─▶ 페이지캐시 ─▶ GPU 폴트(4 KiB)            복사 1, 25배 느림
gTier            NVMe ──DMA──▶ cudaHostAllocMapped ──▶ GPU        복사 0
```

`cudaHostAlloc(cudaHostAllocMapped)` 메모리는 **동시에** (a) O_DIRECT로 NVMe 컨트롤러가 DMA하는
대상이고 (b) GPU가 직접 주소 지정하는 메모리다. **드라이브가 GPU가 읽을 메모리에 직접 쓴다.**
discrete GPU에서는 GDS 없이 불가능하고, Thor에는 GDS가 없다.

### 3.2 API — 엔진이 아니라 I/O 계층

```c
gtier *g = gtier_open(path, &cfg);
gtier_fetch(g, ranges, n, dev_out);   // 흩어진 바이트 범위 → 디바이스 포인터
```

클라이언트는 out-of-core GPU 워크로드 일반이다 — LLM 가중치 스트리밍, 양자 상태벡터,
벡터 인덱스, DB 버퍼풀. 헤더는 [`lib/gtier.h`](lib/gtier.h).

### 3.3 Read planner — 두 규칙 모두 측정에서 도출

**규칙 1 — 고립된 범위를 증폭하지 마라.** O_DIRECT 경계(4 KiB)까지만 정렬한다. 장치의
대역폭-크기 곡선이 선형보다 느리게 자라므로 증폭이 언제나 이득을 앞지른다(§4.3).

**규칙 2 — 인접 범위도 병합하지 마라.** 깊은 큐에서 요청은 거의 공짜이고 바이트만 값을 한다.
얕은 큐에서도 이 장치의 요청 고정비용이 증폭 비용보다 작다(§4.4).

**보정은 모델이 아니라 측정으로.** 파라메트릭 모델(고정 요청비용 + 바이트당 비용)로 임계값을
유도했다가 틀렸다 — 이 장치의 대역폭-크기 곡선은 선형도 단조도 아니다(랜덤 256 KiB가 1.179 GiB/s로
64 KiB의 4.311보다 느리다). 현재는 planner 자체를 후보 임계값들로 돌려보고 최선을 고른다.

### 3.4 상주 정책

캐시는 고정 크기 블록을 요구하고, 그것은 규칙 1이 금지하는 증폭이다. **재사용이 있으면 상환되므로
교차점이 존재한다.** 그리고 그 교차점은 240배로 날카롭다(§4.5).

- `NONE` — 모든 범위를 정확히 인출. 재사용 없음, 증폭 없음
- `BLOCK` — 전부 고정 블록 경유. 재사용이 증폭을 갚지만, miss는 값만 치른다
- `HYBRID` — 미스를 정확 인출하고 N회 접근 시 승격. **실패했다**(§4.5)
- `ADAPTIVE` — **히트율을 관측해 체제를 전환**. 히스테리시스 40%/70%

---

## 4. 평가

모든 측정: 실데이터(비희소·비압축) 파일, 실행마다 페이지 캐시 비움, `jetson_clocks` 고정,
백엔드마다 별도 프로세스.

### 4.1 3계층 전제의 반증

`-ngl K`는 discrete GPU에서 몇 개 레이어가 VRAM에 상주하고 몇 개가 토큰마다 스트리밍되는지를
결정하며 처리량과 PCIe 트래픽을 **동시에** 지배한다.

| 모델 | 처리량 변화 (`-ngl` 0→전체) | 디스크 읽기 변화 |
|---|---:|---:|
| Qwen2.5-7B Q8 (7.5 GiB) | **2.94x** | **+0.3%** |
| Qwen2.5-14B Q8 (14.6 GiB) | **2.70x** | **+0.2%** |
| Qwen2.5-32B Q8 (32.4 GiB) | **2.06x** | **+0.14%** |

> 처리량은 최대 2.94배 바뀌지만 **추가로 이동하는 바이트는 측정 오차 안에서 0이다.**
> `-ngl`은 이 하드웨어에서 데이터 배치가 아니라 **연산 배치** 손잡이다.

상세: [`results/E1_NGL_SWEEP.md`](results/E1_NGL_SWEEP.md)

### 4.2 원인 — readahead가 GPU 폴트에 닿지 않는다

같은 파일, 같은 매핑, 같은 접근 패턴. **누가 폴트하느냐만** 다르다.

| 폴트 주체 | 처리량 |
|---|---:|
| **GPU 전체** | **0.220 GiB/s** |
| CPU 스레드 **1개** | **2.736 GiB/s** |
| CPU 4개 | 2.886 GiB/s |

**CPU 스레드 하나가 GPU 전체보다 12.4배 빠르다.** mmap도, 4 KiB 페이지 자체도, 스토리지도
원인이 아니다. 커널 readahead가 CPU 폴트는 큰 I/O로 묶어주지만 GPU 폴트 경로에는 닿지 않는다.

**메커니즘이 입도임을 같은 시스템에서 확정했다** — gTier의 슬롯 입도만 바꾸면:

| 슬롯 | 4 KiB | 16 KiB | 64 KiB | 256 KiB | **1 MiB** | 256 MiB |
|---|---:|---:|---:|---:|---:|---:|
| GiB/s | **0.198** | 0.681 | 2.144 | 4.373 | **5.442** | 5.369 |

4 KiB를 주면 gTier도 OS 경로(0.240)와 똑같이 붕괴한다.

상세: [`results/GRANULARITY.md`](results/GRANULARITY.md)

### 4.3 입도 바닥

희소 워크로드는 크기 `g`의 항목이 흩어져 있다. 정확히 가져오거나, 주변을 크게 가져온다.
애플리케이션이 얻는 것은 **유효 대역폭** = (항목 수 × g) / 경과시간.

| g | 최적 블록 | 최적 유효 대역폭 | 1 MiB 대비 |
|---:|---:|---:|---:|
| 4 KiB | 4 KiB | 0.530 GiB/s | **0.12x** |
| 16 KiB | 32 KiB | 1.410 | 0.33x |
| **64 KiB** | 64 KiB | 3.146 | 0.73x |
| 256 KiB | 256 KiB | 3.224 | 0.75x |
| 1 MiB | 1 MiB | **4.324** | 1.00x |

**증폭은 거의 언제나 손해다.** g=4 KiB에서 64 KiB 블록으로 키우면 raw가 8.2배 오르지만
증폭이 16배라 유효 대역폭은 **0.51배로 악화**한다. 유일한 예외가 16→32 KiB(정확히 상쇄)이고
그것이 교차점이다. *"큰 블록으로 읽어 대역폭을 얻어라"* 는 순차 접근에서만 맞다.

**함의:** miss가 플래시로 갈 때 희소성 활용 입도에 상한이 생긴다.

| 시스템 | 희소성 단위 | 크기 | 유효 대역폭 | |
|---|---|---:|---:|---|
| MoE-Infinity, Klotski, Pre-gated MoE | 전문가 | 수십~수백 MiB | 4.3 GiB/s | ✅ |
| FlexGen, ZeRO-Infinity | 레이어 | 수백 MiB | 4.3 GiB/s | ✅ |
| **PowerInfer** | 뉴런 | 수~수십 KiB | **0.53~1.41** | ❌ 3~8배 손실 |
| **InfiniGen, vLLM** | KV 페이지 | 4~64 KiB | **0.53~3.1** | ❌ |

상세: [`results/GRANULARITY_FLOOR.md`](results/GRANULARITY_FLOOR.md)

### 4.4 병합하지 마라

인접 범위(간격 = 항목 크기), 64 KiB × 32:

| 임계값 | 요청/인출 | 증폭 | 유효 대역폭 |
|---|---:|---:|---:|
| **병합 없음** | 32.0 | 1.00x | **2.054 GiB/s** |
| 64 KiB | 4.0 | 1.88x | 1.908 (−7%) |
| 1 MiB | 4.0 | 1.88x | 1.752 (−15%) |

얕은 큐(n=4)에서도 1.046 → 0.782 (−25%). 요청을 32→4로 줄여도 증폭 1.88배를 이기지 못한다.

### 4.5 상주 — 체제 감지기가 블록 휴리스틱을 이긴다

64 KiB 항목, 1 MiB 블록(증폭 16x), 윈도우 64 블록:

| 워킹셋 | exact | block | HYBRID | **ADAPTIVE** |
|---:|---:|---:|---:|---:|
| 8 | 3.749 | 78.144 | 80.62 | **82.395** |
| 32 | 2.858 | 80.411 | 78.57 | **81.484** |
| 64 | 2.708 | **81.221** | 25.85 | 80.883 |
| 96 | **2.712** | 0.331 | 1.09 | 2.306 |
| 128 | **2.717** | 0.330 | 0.87 | 2.340 |
| 512 | **2.717** | 0.335 | 0.65 | 2.323 |
| 1024 | **2.551** | 0.284 | — | 2.205 |

**순수 정책은 각각 한쪽 끝에서 7~8배 진다.** block은 워킹셋이 넘칠 때 exact 대비 8.2배 손해,
exact는 들어갈 때 block 대비 29.9배 손해.

**HYBRID는 실패했다.** "N회 접근 시 승격"이 *곧 재사용* 과 *언젠가 재사용* 을 구분하지 못한다 —
워킹셋이 윈도우의 8배면 모든 블록이 결국 두 번 접근되지만 상주 중에는 아니다.

**ADAPTIVE가 성공했다.** 두 순수 정책이 **반대 방향으로** 실패하고 경계가 240배로 날카로우니,
문제는 *어떤 블록을 승격할까* 가 아니라 *지금 어느 체제인가* 다. 전 구간에서 최선의 15% 이내.

상세: [`results/BACKENDS.md`](results/BACKENDS.md)

### 4.6 값을 하지 않는 세 가지 (음성 결과)

**CPU 작업은 공짜다.** 배경 CPU 부하가 114.5 GiB/s를 소비해도 gTier 처리량이 변하지 않는다
(5.273 → 5.391). out-of-core의 병목은 메모리가 아니라 스토리지이고 메모리 여유가 45배다.

**GPU/IO 중첩은 값을 하지 않는다.** 2단 파이프라인을 모든 백엔드에 적용해 0.95~1.05x.
연산 강도를 올리면 교차점이 **~1000 FLOPs/byte**로 드러나는데, LLM 디코드는 **2~4**,
상태벡터 시뮬레이션은 **0.38**이다 — 세 자릿수 아래다. FlexGen의 기여가 GPU·CPU·디스크 I/O를
겹치는 스케줄인데, **이 플랫폼에서 그 차원의 가치는 0이다.**

**전송 메커니즘은 작은 입도에서만 값을 한다.** §4.7 참조.

상세: [`results/PIPELINE.md`](results/PIPELINE.md)

### 4.7 백엔드 비교

여섯 경로가 같은 요청에 답한다. `pread+copy`는 FlexGen/ZeRO-Infinity 패턴,
`uvm`은 DeepUM 패턴, `cufile`은 NVIDIA GDS다.

**합성 워크로드** (인출당 4 MiB, 흩어진 접근, GiB/s):

| item | **gtier** | cufile | pread+copy | mmap-cpu | uvm | mmap-gpu | vs cuFile |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 KiB | **1.817** | 0.650 | 0.658 | 0.114 | 0.125 | 0.040 | **2.80x** |
| 64 KiB | **3.281** | 1.720 | 1.609 | 0.310 | 0.356 | 0.120 | **1.91x** |
| 256 KiB | **4.059** | 2.782 | 2.645 | 0.847 | 0.812 | 0.299 | **1.46x** |
| 1 MiB | **5.084** | 3.960 | 3.695 | 1.368 | 1.327 | 0.727 | **1.28x** |
| 4 MiB | **5.154** | 4.459 | 4.251 | 3.218 | 1.729 | 1.536 | **1.16x** |

연속 제출(`gtier_submit`/`gtier_wait`) 적용 후 수치다. 동기식 인출은 배치 사이에 큐를 비워
얕은 깊이에서 최대 31.7%를 잃었고, 그 때문에 이전 판에서는 1 MiB에서 cuFile에 졌다.
상세: [`results/ASYNC_FIX.md`](results/ASYNC_FIX.md)

**실제 GGUF 가중치 트레이스** (Qwen2.5-32B Q8_0, 32.42 GiB, 9 샤드, 64 레이어, 콜드 캐시):

| 백엔드 | GiB/s | tok/s |
|---|---:|---:|
| **gtier** | **5.105** | **0.1575** |
| cufile | 5.025 | 0.1550 |
| pread+copy | 4.641 | 0.1431 |
| mmap-cpu | 4.350 | 0.1342 |
| uvm | 2.644 | 0.0816 |
| **mmap-gpu** | 1.448 | 0.0447 |

실제 트레이스 입도별(GiB/s): 64 KiB에서 gtier 3.497 / cufile 3.089 / mmap-gpu 2.258,
1 MiB에서 5.135 / 4.321 / 1.474. **gTier가 모든 입도에서 앞서며 cuFile 대비 1.13~1.27x,
OS 경로 대비 1.55~3.48x.**

---

## 5. 정직한 한계

- **전송 메커니즘의 우위는 입도에 의존한다.** 연속 제출을 고친 뒤 합성 워크로드에서 cuFile 대비
  1.16~2.80x이며 작은 입도일수록 크다. 그래도 상주(240x)와 입도(10~20x)가 더 큰 레버다.
- **절대 성능이 이 워크로드의 결론이다.** 32 GiB 모델에서 **0.157 tok/s**, 토큰당 6.4초.
  dense 모델의 가중치 스트리밍은 이 플랫폼에서 현실적이지 않다. MoE의 희소성이 전제다.
- **탑티어 베이스라인을 실제로 돌리지 못했다.** MoE-Infinity, PowerInfer, FlexGen 비교에는
  sm_110용 PyTorch가 필요한데 PyPI aarch64 휠은 CPU 전용이다(`2.9.1+cpu`, arch 리스트 비어 있음).
  현재 비교는 이 저장소가 직접 구현한 다섯 경로에 한정된다.
- **DRAM을 초과하는 모델을 아직 측정하지 못했다.** Qwen3-235B-A22B Q4_K_M(132.4 GiB, DRAM의
  1.09x) 다운로드가 진행 중이다. cgroup으로 에뮬레이션하려 했으나 Tegra의 `cudaMalloc`이
  memory cgroup 회계를 빠져나가 유효하지 않다([`results/E2_PARTIAL.md`](results/E2_PARTIAL.md)).
- **`gtier_fetch`는 동기식이다.** §4.6이 보인 대로 이 워크로드에서는 문제가 되지 않지만,
  연산 강도가 높은 클라이언트에는 비동기 API가 필요하다.
- **초기 커밋의 "비결정적 붕괴" 주장은 철회했다.** `ftruncate`가 만든 희소 파일에서 측정한
  아티팩트였다([`results/REGIMES.md`](results/REGIMES.md)).

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
│   ├── gguf.h / gguf.c          GGUF 파서 (실제 모델 트레이스용)
│   ├── bench.cu                 합성 워크로드 드라이버
│   ├── gguf_bench.cu            실제 가중치 트레이스 드라이버
│   ├── run_bench.sh             백엔드별 캐시 비움 드라이버
│   └── run_pipe.sh              파이프라이닝 비교 드라이버
├── gtier/                       초기 프로브 (마이크로벤치마크)
│   ├── gtier.cu                 OS 경로 vs 윈도우 비교
│   ├── randread.c               NVMe 입도별 랜덤·순차
│   ├── sparsefetch.c            입도 바닥 전면 측정
│   └── cufile_test.cu           GDS 가용성 검증
├── mmap_probe/ · bw_probe/      GPU mmap regime · CPU+GPU 대역폭
├── bench/                       LLM 엔진 측정 하네스 (llama.cpp)
├── scripts/                     모델 다운로드
└── results/                     모든 측정 결과
    ├── GRANULARITY.md           §4.2 원인 분리와 입도 메커니즘
    ├── GRANULARITY_FLOOR.md     §4.3 입도 바닥
    ├── BACKENDS.md              §4.4~4.5, §4.7 백엔드·상주 정책
    ├── PIPELINE.md              §4.6 파이프라이닝
    ├── E1_NGL_SWEEP.md          §4.1 3계층 반증
    ├── E2_PARTIAL.md            DRAM 초과 (부분, 방법론 한계 포함)
    ├── LIBRARY.md               planner 설계 기록
    └── REGIMES.md               철회된 초기 측정 (기록 보존)
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
```

다른 Jetson 세대에서는 `ARCH`를 바꾼다.

```bash
make ARCH=sm_87           # Orin
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
cfg.max_fetch_ranges = 32;             // 한 번에 낼 최대 범위 수

gtier *g = gtier_open("model.gguf", &cfg);

gtier_range r[32] = { {off0, len0}, {off1, len1}, /* ... */ };
void *dev[32];
gtier_fetch(g, r, 32, dev);            // dev[i] 는 GPU가 바로 읽는 포인터

my_kernel<<<...>>>((const uint8_t *)dev[0], len0);

gtier_stats s; gtier_get_stats(g, &s);
gtier_close(g);
```

**제약.** 한 범위는 하나의 슬롯 안에 들어가야 한다. O_DIRECT 정렬이 양끝으로 최대 4 KiB씩
늘어나므로 범위 길이를 `slot_bytes - 8192` 이하로 유지한다. 캐시 정책을 쓸 때는 범위가 블록
경계를 걸치지 않아야 한다. 위반 시 `-E2BIG`을 돌려준다.

링크:

```bash
nvcc -O3 -std=c++17 -arch=sm_110 myapp.cu libgtier.a -luring -lcufile \
  -I/usr/local/cuda/targets/sbsa-linux/include \
  -L/usr/local/cuda/targets/sbsa-linux/lib -o myapp
```

### 7.4 측정 재현

측정 전 클럭을 고정한다.

```bash
sudo jetson_clocks
nvpmodel -q                # 전력 모드 기록
```

**주의:** 테스트 파일은 반드시 **실제 데이터**여야 한다. `ftruncate`나 `truncate`로 만든 희소
파일은 구멍을 읽을 때 커널이 디스크를 건드리지 않고 제로 페이지를 할당하므로, 스토리지가 아니라
페이지 할당을 측정하게 된다.

```bash
# 비압축 실데이터 32 GiB
openssl enc -aes-256-ctr -pass pass:gtier -nosalt < /dev/zero 2>/dev/null \
  | dd of=/tmp/real32.bin bs=64M count=512 iflag=fullblock oflag=direct
du -h --apparent-size /tmp/real32.bin && du -h /tmp/real32.bin   # 둘이 같아야 함
```

백엔드 비교 (백엔드마다 페이지 캐시를 비운다):

```bash
cd lib
./run_bench.sh --file /tmp/real32.bin --item 65536 --n 64 --iters 64
```

상주 정책 비교:

```bash
./gtier_bench --file /tmp/real32.bin --item 65536 --n 64 --slots 64 \
  --policy 3 --reuse 128 --iters 256 --only 0     # policy 3 = ADAPTIVE
```

파이프라이닝:

```bash
./run_pipe.sh --file /tmp/real32.bin --item 1048576 --n 16 --iters 64 --policy 0
```

실제 모델 가중치 트레이스:

```bash
./gguf_bench $(for f in /path/to/model/*.gguf; do echo --shard $f; done) \
  --backend 0 --policy 0 --tokens 1 --slot 1048576 --slots 64
```

`--backend` 는 0=gtier, 1=mmap-gpu, 2=mmap-cpu, 3=pread+copy, 4=cufile, 5=uvm.
`--policy` 는 0=NONE, 1=BLOCK, 2=HYBRID, 3=ADAPTIVE.

모델 받기:

```bash
./scripts/fetch.sh      # Qwen2.5 dense 사다리 (7B/14B/32B Q8_0)
./scripts/fetch2.sh     # Qwen3-235B-A22B MoE (Q3_K_M, Q4_K_M)
```

### 7.5 초기 프로브 (선택)

§4.2와 §4.3의 원측정을 그대로 재현하려면:

```bash
cd gtier && make
./gtier --file /tmp/real32.bin --gib 32 --mode mmap        # OS 경로
./gtier --file /tmp/real32.bin --gib 32 --mode mmap-cpu    # CPU가 폴트
./gtier --file /tmp/real32.bin --gib 32 --mode gtier --slot-kib 1024
gcc -O2 -o sparsefetch sparsefetch.c -luring
./sparsefetch /tmp/real32.bin 65536 65536 512 64           # 입도 바닥
```
