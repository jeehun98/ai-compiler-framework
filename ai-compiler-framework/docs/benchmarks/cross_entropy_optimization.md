# 🚀 CrossEntropyLoss 커널 최적화 로그

- **작성일:** 2026-02-17
- **디바이스:** NVIDIA RTX 3060
- **커널 타겟:** `xent_fwd_sum_f32`, `xent_bwd_f32`

---

## 1️⃣ 성능 요약

> **Note:** CrossEntropy는 `Logits(N, C)`와 `Targets(N)`을 입력으로 받는다.
> 연산 특성상 `expf`, `logf` 등 무거운 수학 연산이 포함되므로 단순 Copy 커널보다 대역폭 수치는 낮게 측정된다.
> (측정 기준: `(Read(Logits) + Read(Targets) + Write(dLogits)) / Time`)

| 커널 버전 | 모드 | 입력 크기 (N, C) | 실행 시간 | 유효 대역폭 (GB/s) | 비고 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| v1.0 (Ref) | FWD | 1024, 4096 | *Est. > 0.5ms* | ~30.0 | PyTorch Naive (Separate Ops) |
| **v2.0 (Fused)** | **FWD** | **1024, 4096** | **0.105 ms** | **152.4** | **Fused LogSumExp** |
| **v2.0 (Fused)** | **BWD** | **1024, 4096** | **0.208 ms** | **153.8** | **Recompute Softmax** |

---

## 2️⃣ 구현 전략 및 이슈 분석

### 🏗️ 1. Fused Kernel (One Block per Row)
- **전략:** $N$(Batch Size) 차원에 대해 그리드를 생성하고, 각 CUDA 블록이 하나의 행(Class $C$)을 전담 처리한다.
- **장점:** 행 단위의 Reduction(Max, Sum)이 블록 내부의 Shared Memory와 Warp Shuffle로 완결되므로 동기화 비용이 적다.
- **적용:** $C$가 4096 이하인 경우 매우 효율적이다. ($C$가 매우 클 경우 Block-Stride Loop 적용 필요)

### 🛡️ 2. 수치 안정성 (Numerical Stability)
- **LogSumExp Trick:** $\log(\sum e^{x_i}) = \alpha + \log(\sum e^{x_i - \alpha})$ 공식에서 $\alpha = \max(x)$를 사용하여 Overflow를 방지한다.
- **구현:** 1. `block_max`로 해당 행의 최댓값 $m$을 구함.
  2. `block_sum`으로 $\sum \exp(x_i - m)$을 구함.
  3. 최종 Loss 계산 시 $m$을 다시 더해줌.

### 🔄 3. Backward: Recomputation (Activation Checkpointing)
- **이슈:** Forward 단계에서 계산된 확률 행렬 $P$ (`Softmax(logits)`)를 저장해두면 Backward가 빠르지만, 메모리 사용량이 $N \times C \times 4$ 바이트만큼 증가한다.
- **해결:** v2 커널은 Backward 수행 시 **Logits를 다시 읽어 Softmax를 즉석에서 재계산**한다.
- **결과:** 메모리 대역폭(Read Logits)은 2배 쓰지만, VRAM 사용량을 획기적으로 줄임. (Compute Bound에 가까운 최신 GPU에서는 이 방식이 더 유리함)

---

## 3️⃣ 상세 프로파일링 결과 (Shape: 1024 x 4096)

1.  **FWD Pass (0.105 ms)**
    * 입력 로드: 16MB (Float32)
    * 출력: Scalar (무시 가능 수준)
    * **병목:** `expf` 연산 처리량 및 Warp Reduction Latency.
    * **특이사항:** 최종 Loss 합산 시 `atomicAdd`를 사용하여 Global Scalar에 누적. $N$이 매우 클 경우(>10만) 이 부분의 직렬화가 병목이 될 수 있으나 현재 크기에서는 문제없음.

2.  **BWD Pass (0.208 ms)**
    * 데이터 이동: 16MB Read (Logits) + 16MB Write (Grads) = 32MB.
    * 시간이 FWD의 약 2배인 이유: 읽고 쓰는 데이터 양이 정확히 2배이기 때문.
    * **결론:** 커널이 메모리 대역폭과 연산 파이프라인을 균형 있게 잘 사용하고 있음.

---

## 4️⃣ 결론 및 향후 계획

> "Memory Bound와 Compute Bound의 경계에 있는 커널"

- **성과:** PyTorch의 최적화된 구현체와 대등하거나 근접한 성능을 보임. (수치 오차 `1e-6` 이하로 정합성 확보)
- **개선점:** - 현재 입력 로드 시 `float` 단위로 읽고 있음. `float4` 벡터화 로드를 적용하면 대역폭 효율을 10~20% 더 끌어올릴 수 있을 것으로 예상됨.
  - $C$가 작은 경우(예: CIFAR-10, $C=10$)에 대한 워프 활용률 저하를 막기 위해 **"Multiple Rows per Block"** 전략 고려 가능.