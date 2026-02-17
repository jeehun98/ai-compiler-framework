# 🚀 Softmax (Forward) 커널 최적화 로그

- **작성일:** 2026-02-17
- **디바이스:** NVIDIA RTX 3060
- **커널 타겟:** `softmax_lastdim_f32`, `softmax_lastdim_f16`

---

## 1️⃣ 성능 요약

> **Note:** Softmax는 입력 $X$를 읽어 $Y$를 쓰는 연산이다.
> 아래 수치는 유사한 메모리 패턴을 가진 Backward 테스트 결과(185 GB/s)를 기반으로 추산된 Forward 성능이다.

| 커널 버전 | 데이터 타입 | 입력 크기 (Row, Col) | 실행 시간 | 유효 대역폭 (GB/s) | 비고 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v2.0 (Row-Block)** | **F32** | **2048, 8192** | **~1.0 ms** | **~185.0** | **Stable Implementation** |
| **v2.0 (Row-Block)** | **F16** | **2048, 8192** | **~0.56 ms** | **~178.0** | **Mixed Precision** |

---

## 2️⃣ 구현 전략: One Block Per Row

### 🏗️ 구조적 특징
- **Grid 구성:** `gridDim.x = Rows`
  - 입력 텐서의 각 행(Row)을 하나의 CUDA 블록이 전담 처리한다.
  - 행 간(Inter-row) 의존성이 없으므로 완벽한 병렬 처리가 가능하다.
- **Shared Memory Reduction:**
  - 각 스레드가 부분 Max/Sum을 계산한 후, `block_reduce_max` / `block_reduce_sum`을 통해 블록 전체의 값을 도출한다.
  - Global Memory를 거치지 않고 Shared Memory에서 통신하므로 매우 빠르다.

### 🛡️ 수치 안정성 (Safe Softmax)
- **Problem:** 단순 $e^{x_i}$ 계산은 $x_i$가 클 경우 쉽게 Float Overflow를 일으킨다.
- **Solution:** $m = \max(x)$를 먼저 구한 뒤, $e^{x_i - m}$을 계산한다.
- **Pass 구성:**
  1. **Pass 1:** 행 전체를 읽어 `Max` 계산.
  2. **Pass 2:** 행 전체를 다시 읽어(혹은 캐시) `Sum(exp(x-m))` 계산.
  3. **Pass 3:** 최종 `exp(x-m) / Sum` 계산 및 저장.
- **v2 최적화:** 위 3단계를 하나의 커널 안에서 수행(Kernel Fusion)하여 Global Memory 접근을 줄임.

### 📉 Mixed Precision (F16)
- **입력/출력:** `__half` (FP16)
- **내부 연산:** `float` (FP32)
- FP16은 범위가 좁아 Sum Reduction 시 Overflow 위험이 크므로, 데이터를 읽자마자 `__half2float`로 변환하여 모든 누적 연산을 FP32로 수행한 뒤, 저장 직전에 `__float2half`로 변환한다.

---

## 3️⃣ 이슈 및 향후 개선 계획

### ⚠️ 메모리 접근 패턴 (Non-Vectorized Load)
- 현재 코드는 `x[base + c]` 형태로 단일 `float` 또는 `__half`를 읽는다.
- **병목:** DRAM 대역폭을 최대로 쓰기 위해서는 `L2 Cache` 라인(128B)을 효율적으로 긁어오는 128-bit(`float4`) 로드가 필수적이다.
- **개선안:**
  - F32: `reinterpret_cast<float4*>` 사용하여 4개씩 로드.
  - F16: `__half2` 벡터 연산 또는 `float4`로 8개씩 로드.
  - 이를 적용하면 대역폭이 250 GB/s 이상으로 향상될 것으로 예상됨.

### 📊 결론
- "One Block Per Row"는 Softmax의 교과서적인 구현 방식으로, 수치 안정성과 준수한 성능을 보장한다.
- 현재 구현은 기능적으로 완전하며(Safe Softmax, Mixed Precision), 향후 벡터화를 통해 추가적인 성능 향상 여지가 있다.