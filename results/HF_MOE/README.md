# 같은 모델·같은 파일·같은 예산: Qwen3-30B-A3B

이 디렉터리는 **MoE-Infinity가 읽는 바로 그 safetensors 파일** 위에서 gTier와
기존 백엔드들을 모두 돌린 결과다. 포맷을 맞추기 위해 모델을 변환하지 않았다.
gTier는 바이트 범위를 읽는 계층이라 포맷을 보지 않으므로, GGUF에 쓰던 트레이스
드라이버를 safetensors에 그대로 겨눌 수 있다(`lib/safetensors.c`).

## 1. 워크로드

| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen3-30B-A3B` (`Qwen3MoeForCausalLM`) |
| 파일 | safetensors 16 샤드, 57 GiB, bf16 |
| 층 | 48 |
| 전문가 | 층당 128, 토큰당 8 활성 (6.25%) |
| 전문가 텐서 | `gate/up/down_proj` 각 3 MiB → 전문가 1개당 9 MiB |
| 토큰당 가중치 | 48 × 8 × 9 MiB ≈ **3.4 GiB** |
| 라우팅 | Zipf skew 0.8, **토큰마다 재추첨** |
| 증폭 | 1.00x (원하는 바이트만 읽음) |

HF 포맷은 전문가를 텐서 하나씩 저장한다(GGUF는 층별로 stack). 따라서 라우팅된
읽기가 텐서 하나 통째가 되고, 트레이스는 stacked/per-expert 두 배치를 모두 다룬다.

## 2. 측정 프로토콜에서 고친 두 가지

두 가지를 바로잡기 전의 숫자는 저장장치가 아니라 RAM을 재고 있었다.

**(a) 토큰마다 재라우팅.** 라우팅을 실행당 한 번만 뽑으면 토큰 1이 읽은 전문가를
토큰 2~8이 그대로 다시 읽는다. 그 결과는 페이지 캐시에서 나오므로 mmap 백엔드가
장치 천장(5.682 GiB/s)을 넘는 10~12 GiB/s를 기록했다. 층마다 토큰마다 다시
뽑도록 바꾸자 mmap-cpu는 12.59 → 5.81 GiB/s로 내려왔다.

**(b) 메모리 예산 동일화.** mmap과 pread는 페이지 캐시 전체(122 GiB)를 공짜
캐시로 쓴다. gTier가 명시적으로 2 GiB 윈도우만 쓰는 것과 비교하면 캐시 크기를
비교하는 셈이다. 그래서 모든 실행을 같은 `memory.max` cgroup 안에 넣었다.

```bash
sudo systemd-run --scope -p MemoryMax=8G -- ./lib/gguf_bench ...
```

같은 mmap-cpu가 예산에 따라 **0.607 → 5.289 GiB/s (8.7배)** 로 움직인다.
예산을 고정하지 않은 저장장치 비교는 이 한 축만으로 결론이 뒤집힌다.

## 3. 결과: 예산 8 GiB, 윈도우 2 GiB, 8 토큰

| 백엔드 | 대응 선행연구 | GiB/s | gTier 대비 |
|---|---|---|---|
| **gTier + async** | — | **4.748** | — |
| **gTier** | — | 4.321 | 0.91x |
| mmap-gpu | — | 3.873 | 0.82x |
| pread+copy | FlexGen, ZeRO-Infinity | 3.701 | 0.78x |
| UVM | DeepUM | 2.554 | 0.54x |
| mmap-cpu | — | 0.607 | 0.13x |
| cuFile | NVIDIA GDS | **실행 불가 (OOM)** | — |

## 4. cuFile은 8 GiB 예산에서 돌지 못한다

윈도우는 2 GiB로 똑같이 주었는데도 cuFile 백엔드만 cgroup OOM으로 죽는다.

```
Memory cgroup out of memory: Killed process (gguf_bench)
  total-vm:24302452kB  anon-rss:8343340kB
```

예산을 올려가며 찾은 경계는 **8 GiB 실패 / 9 GiB 성공**이다.

| 예산 | cuFile |
|---|---|
| 8 GiB | OOM |
| 9 GiB | 3.815 GiB/s |
| 10 GiB | 3.835 GiB/s |
| 11 GiB | 3.839 GiB/s |
| 12 GiB | 3.817 GiB/s |

Thor에는 GDS 커널 경로가 없다(`nvfs major=0 minor=0`, `dstatusflags=0x0`).
cuFile은 POSIX 호환 모드로 내려가 바운스 버퍼를 통해 복사하는데, 그 버퍼는
호출자가 요청한 윈도우에 계산되지 않는다. 즉 **같은 2 GiB 윈도우를 선언해도
실제로는 8 GiB를 상주시킨다.**

### 상주 메모리 (3 토큰, 예산 40 GiB, `/usr/bin/time -v`)

| 백엔드 | 선언 윈도우 | 실제 maxRSS | 배수 | GiB/s |
|---|---|---|---|---|
| **gTier** | 2.00 GiB | **2.08 GiB** | **1.04x** | 4.353 |
| **gTier + async** | 2.00 GiB | **2.08 GiB** | **1.04x** | **4.792** |
| pread+copy | 2.00 GiB | 2.08 GiB | 1.04x | 3.711 |
| cuFile | 2.00 GiB | 8.09 GiB | 4.04x | 3.852 |
| UVM | 2.00 GiB | 0.69 GiB | 0.35x | 2.752 |
| mmap-gpu | 2.00 GiB | 11.98 GiB | 5.99x | 2.670 |
| mmap-cpu | 2.00 GiB | 11.62 GiB | 5.81x | 5.289 |

UVM만 윈도우보다 적게 쓰는데, 페이지를 드라이버가 관리하므로 사용자 공간 RSS에
잡히지 않기 때문이다. 대신 대역폭이 가장 낮다(2.752).

### cuFile이 돌 수 있는 예산(12 GiB)에서 다시 비교

| 백엔드 | GiB/s |
|---|---|
| gTier + async | **4.794** |
| gTier | 4.338 |
| cuFile | 3.839 |

cuFile에게 필요한 만큼 예산을 다 주고 비교해도 **+24.9%** 앞선다.
8 GiB에서는 비교 자체가 성립하지 않는다.

## 5. 캐시 정책 (예산 8 GiB, 윈도우 2 GiB = 모델의 3.5%)

| 정책 | GiB/s | 적중률 | 증폭 |
|---|---|---|---|
| NONE | 4.321 | 0.0% | 1.00x |
| PIN | 4.133 | 5.4% | 0.97x |
| BLOCK (LRU) | 3.547 | 47.2% | 1.24x |
| ADAPTIVE | 3.562 | 47.2% | 1.24x |

**적중률 47.2%인데 더 느리다.** 블록 캐시는 4 MiB 단위로 담는데 전문가 텐서는
3 MiB이고 정렬이 맞지 않아 증폭 1.24x가 붙는다. 24% 더 읽어서 47% 맞히면
남는 게 없다. 이 워크로드에서 윈도우가 모델의 3.5%뿐이고 skew 0.8의 hot set이
윈도우보다 훨씬 크다는 점도 같이 작용한다.

정직한 결론: **이 설정에서 캐싱은 도움이 되지 않는다.** 도움이 되는 것은
비동기 파이프라인(+9.9%)이다.
