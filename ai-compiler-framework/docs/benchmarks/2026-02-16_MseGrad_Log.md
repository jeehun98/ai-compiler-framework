# 📉 MseGrad 커널 벤치마크 (Loss Function)

- **작성일:** 2026-02-16
- **디바이스:** NVIDIA RTX 3060
- **커널 타겟:** mse_grad_f32, mse_grad_f16_vec2

---

## 1️⃣ 성능 요약 (유효 대역폭)

> Purpose: 손실 함수(Loss)의 Gradient 계산. Backpropagation 시작점.  
> Pattern: Read(Pred) + Read(Target) + Write(Grad) → **2 Read + 1 Write**  
> 유효 대역폭 정의: **(bytes(pred) + bytes(target) + bytes(grad)) / time**

| 데이터 타입 | 입력 크기 | 실행 시간 | 유효 대역폭 (GB/s) | 효율(vs Copy F32) | 비고 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| F32 | 4096² (16M) | 0.748 ms | **269.29** | 90% | Very Good |
| F16 Naive | 4096² (16M) | 0.383 ms | 262.92 | 88% | 스칼라 로드 |
| F16 Vec2 | 4096² (16M) | 0.377 ms | **267.28** | 90% | 벡터화 |

---

## 2️⃣ 분석: Pure Memory Bound

### ⚖️ Naive vs Vec2 차이가 작은 이유
- F16 Naive(262.9) vs Vec2(267.3)의 차이는 약 1~2%로 작다.
- 이는 커널이 수행하는 `sub`/`mul` 연산보다 **메모리 이동(2 Read + 1 Write)** 이 지배적이기 때문이다.
- 즉, 명령 발행 최적화(벡터화)보다 **대역폭/스토어 경로**가 전체 성능을 제한한다는 신호로 해석된다.

### 📉 Add(≈314 GB/s) 대비 낮은 이유
- Add와 달리 MseGrad는 **2 Read + 1 Write** 패턴이며, 스토어 포함 트래픽 특성상 실효 대역폭이 더 낮게 관측될 수 있다.
- 그럼에도 Copy baseline 대비 90% 수준이면 매우 우수한 구현으로 판단된다.

---

## 3️⃣ 결론
- 약 267~269 GB/s는 Copy baseline 대비 90% 수준의 높은 효율로, 학습 루프에서 병목 가능성이 낮다.
- 벡터화 이득이 작다는 점은 이 커널이 연산이 아니라 메모리(대역폭/스토어 경로)에 의해 지배됨을 뒷받침한다.
