# 📉 BatchNorm 커널 베이스라인 분석 (Performance Bottleneck)

- **작성일:** 2026-02-16
- **디바이스:** NVIDIA RTX 3060
- **커널 타겟:** `batchnorm_fwd_f16_nchw` (Naive Atomic Implementation)

---

## 1️⃣ 벤치마크 결과 요약

> **Status:** ⚠️ **Critical Performance Issue Detected** (Training Path)

| 모드 | 입력 크기 (N,C,H,W) | 실행 시간 | 대역폭 (GB/s) | 상태 |
| :--- | :--- | :--- | :--- | :--- |
| **Fwd Inference** | 32, 64, 128, 128 | 0.733 ms | **183.18 GB/s** | ⚠️ **Sub-optimal** (목표치 300+ GB/s) |
| **Fwd Training** | 32, 64, 128, 128 | 83.058 ms | **2.42 GB/s** | ❌ **Severe Bottleneck** (Atomic Contention) |
| **Bwd Training** | 32, 64, 128, 128 | 82.824 ms | **4.05 GB/s** | ❌ **Severe Bottleneck** (Atomic Contention) |

---

## 2️⃣ 병목 원인 분석: "Global Atomic Contention"

### 🔎 현상
학습(Training) 모드가 추론(Inference) 모드보다 **약 100배** 느리게 측정됨.

### 🔎 원인: NCHW 레이아웃과 Global Atomic의 부조화
1.  **데이터 레이아웃:** NCHW 포맷에서 같은 채널($C$)의 데이터는 $H \times W$ 크기만큼 연속되어 있음.
2.  **스레드 매핑:** 현재 커널은 인접 스레드가 인접 데이터를 처리함.
3.  **충돌 발생:** $H \times W = 128 \times 128 = 16,384$개의 데이터가 연속됨. 즉, **16,384개의 스레드가 동시에** `sum[c]` 주소 하나에 `atomicAdd`를 시도함.
4.  **하드웨어 동작:** GPU의 L2 Cache/DRAM 컨트롤러에서 이 요청들을 직렬화(Serialize)하여 처리하느라 엄청난 지연 발생.

### 🔎 추가 요인: Debug Print
- 벤치마크 루프 내부에서 `fprintf(stderr)`가 호출되어 Host-Device 동기화 및 I/O Latency가 포함됨. (Inference 성능이 183GB/s에 그친 주원인)

---

## 3️⃣ 향후 최적화 계획 (Next Steps)

이 성능 문제를 해결하기 위해 다음 단계의 최적화가 필수적임.

### ✅ Step 1: Block Reduction (Warp Shuffle) 
- **전략:** Global Memory에 바로 쓰지 않고, **레지스터(Warp Shuffle)** 또는 **Shared Memory**를 사용하여 스레드 블록 단위로 먼저 합산(Partial Sum)을 수행.
- **예상 효과:** Atomic 충돌 횟수가 `1/256` ~ `1/1024`로 감소하여 Training 성능이 Inference 수준으로 회복될 것.

### ✅ Step 2: Logging 제거
- 벤치마크 모드에서는 `fprintf`를 비활성화하여 순수 커널 실행 시간만 측정.

### ✅ Step 3: Vectorized Load
- Inference의 경우(183 GB/s)에도 아직 대역폭 포화(310 GB/s)에 도달하지 못함. `half2` 또는 `float4` 벡터 로드를 적용하여 Memory Transaction 효율 개선 필요.

---

## 4️⃣ 결론
현재의 Naive 구현은 **기능적 정합성(Correctness)**은 확보했으나, NCHW 레이아웃에서의 **Atomic 연산 비용**을 간과하여 실사용이 불가능한 수준의 성능을 보임. **Block Reduction 도입이 시급함.**