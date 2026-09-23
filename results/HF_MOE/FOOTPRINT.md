# 메모리 발자국을 어떻게 재야 하는가 — 그리고 앞선 표의 정정

## 1. `window`는 메모리 상한이 아니다

`gtier_config.slot_bytes × slots`를 모든 백엔드에 똑같이 2.00 GiB로 주었지만,
백엔드마다 그 값을 **다르게 쓴다**. 설정을 읽지 않는 백엔드도 있다.

| 백엔드 | 실제 할당 | 윈도우 배수 |
|---|---|---|
| gtier | `cudaHostAlloc(Mapped)` slots×slot | 1배 |
| pread+copy | `cudaHostAlloc` slots×slot **+ `cudaMalloc`** slots×slot | **2배** |
| cufile | `cudaMalloc` slots×slot + `cuFileBufRegister` | 1배 + cuFile 내부 |
| uvm | `cudaMallocManaged` slots×slot | 1배 |
| **mmap-gpu / mmap-cpu** | `mmap(file_bytes)` — **파일 전체** | **설정을 안 씀** |

따라서 출력 줄의 `win=2.00 GiB`는 mmap 백엔드에 대해서는 **의미가 없다**.
그 값은 설정에서 계산된 숫자일 뿐 mmap은 파일 57 GiB를 통째로 매핑한다.

## 2. 세 가지 계측이 모두 다른 것을 센다

통합 메모리 SoC에서 무엇이 무엇을 세는지 직접 측정했다
(프로브: 6 GiB를 할당하고 전 페이지를 건드린 뒤 cgroup 반응을 본다).

| 할당 | cgroup `memory.max` 과금 | RSS 계수 | `cudaMemGetInfo` 계수 |
|---|---|---|---|
| `malloc` | 예 (초과 시 OOM kill) | 예 | 아니오 |
| `cudaHostAlloc(Mapped)` | 예 (OOM kill) | 예 | **예 (중복)** |
| `cudaMallocManaged` | 예 (OOM kill) | 예 | 예 |
| **`cudaMalloc`** | **아니오** | **아니오** | 예 |
| `mmap` 파일 페이지 | 예, 그러나 **회수 가능** (죽지 않고 압박받음) | 예 | 아니오 |

검증: 4 GiB cgroup 안에서 `cudaMalloc` 6 GiB가 **성공한다** (RSS 0.07 GiB).
같은 조건에서 `malloc`·`cudaHostAlloc`·`cudaMallocManaged` 6 GiB는 OOM kill된다.

결과적으로,

- **RSS는 `cudaMalloc`을 빠뜨린다** → 디바이스 버퍼를 쓰는 백엔드를 과소평가한다.
- **`cudaMemGetInfo`는 매핑된 호스트 메모리를 중복 계수한다** → 합산이 불가능하다.
- **cgroup은 디바이스 메모리를 과금하지 않는다** → 예산 실험은 디바이스를 쓰는
  베이스라인에게 **유리한** 쪽으로 편향돼 있다.

## 3. 올바른 계측: `MemAvailable` 하락폭

`MemAvailable`은 회수 가능한 페이지 캐시를 이미 할인한 값이므로, 그 하락폭은
**다른 프로세스가 더 이상 쓸 수 없게 된 메모리**와 정확히 같다.
프로브: [`../../scripts/memfoot.sh`](../../scripts/memfoot.sh)

Qwen3-30B-A3B, 배치 8, 16 토큰, 선언 윈도우 2.00 GiB, 실행마다 캐시 비움.

| 백엔드 | 선언 윈도우 | **실제 점유** | 배수 | GiB/s |
|---|---:|---:|---:|---:|
| **gtier + async** | 2.00 GiB | **2.33 GiB** | **1.17x** | **4.710** |
| gtier | 2.00 GiB | 2.35 GiB | 1.18x | 4.262 |
| pread+copy | 2.00 GiB | 4.35 GiB | **2.18x** | 3.724 |
| cufile | 2.00 GiB | **10.62 GiB** | **5.31x** | 3.820 |
| uvm | 2.00 GiB | 1.29 GiB | 0.65x | 2.542 |
| mmap-gpu | (해당 없음) | 0.54 GiB | — | 2.274 |
| mmap-cpu | (해당 없음) | 0.35 GiB | — | 4.535 |

## 4. 앞선 표에서 정정해야 할 것

이전 `rss.txt`는 RSS만 보았으므로 세 가지가 틀렸다. 그 파일은 삭제했다.

- **철회: "mmap이 윈도우의 6배를 쓴다."** 틀렸다. mmap이 점유한 11.98 GiB는
  **회수 가능한 페이지 캐시**였고, 커널이 압박받으면 돌려준다. 회수 불가능한
  점유는 0.35~0.54 GiB로 전 백엔드 중 가장 작다. mmap의 실제 약점은 메모리
  사용량이 아니라, 예산이 조이면 **대역폭이 무너진다**는 것이다
  (8 GiB 예산에서 mmap-cpu 4.535 → **0.607 GiB/s**).
- **정정: pread+copy는 2.08이 아니라 4.35 GiB.** 호스트 스테이징과 디바이스
  버퍼를 둘 다 잡는데 RSS가 후자를 못 봤다. 윈도우의 **2.18배**다.
- **정정: cuFile은 4.04배가 아니라 5.31배.** 10.62 GiB다.
- **정정: gTier는 1.04배가 아니라 1.17배.** 차이는 CUDA 컨텍스트 비용이며,
  모든 백엔드가 똑같이 낸다.

## 5. 남는 주장

선언한 예산을 지키는 것은 gTier와 UVM뿐이다. UVM은 가장 적게 쓰지만 가장 느리다
(2.542 GiB/s, gTier의 54%). mmap은 적게 점유하지만 예산이 조이면 무너진다.
**대역폭 1위이면서 선언 예산을 지키는 것은 gTier뿐이고, 배수는 1.17x 대
pread+copy 2.18x, cuFile 5.31x이다.**
