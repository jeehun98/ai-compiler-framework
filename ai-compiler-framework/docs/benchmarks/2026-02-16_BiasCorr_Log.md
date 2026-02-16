# 🛠️ BiasCorr (Adam Helper) 커널 로그

- **작성일:** 2026-02-16
- **디바이스:** NVIDIA RTX 3060
- **커널 타겟:** biascorr_f32_v0

---

## 1️⃣ 기능 요약

> **Purpose:** Adam Optimizer의 Bias Correction 계수  
> BC1 = 1 / (1 - β1^t),  BC2 = 1 / (1 - β2^t) 를 계산한다.

| 입력 (Input) | 출력 (Output) | 속성 (Attribute) |
| :--- | :--- | :--- |
| Step (Int32 스칼라) | BC1, BC2 (Float32 스칼라) | β1, β2 (Schema: `BCOR`) |

---

## 2️⃣ 테스트 결과 (정확성)

> 테스트 파라미터: **β1 = 0.9, β2 = 0.999**

| Case | Step Input | Result (BC1 / BC2) | 상태 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| Normal | t=1 | 10.0 / 1000.0 | ✅ Pass | 초기 보정값 |
| Normal | t=10 | 1.53 / 100.4 | ✅ Pass | 지수 감소 확인 |
| Clamp | t=0 | 10.0 / 1000.0 | ✅ Pass | t<1이면 t=1로 처리 (Div/0 회피) |
| Rank | Rank-0 / Rank-1 | 동일 결과 | ✅ Pass | 스칼라/1D 텐서 호환 |

---

## 3️⃣ 성능 특성: Launch-Latency 지배

### 📉 분석
- 이 커널은 GB/s나 TFLOPS로 측정하는 것이 의미 없다.
- 커널 내부 연산은 매우 작으며, 관측 지연의 대부분은 **Kernel Launch Overhead(수 µs 수준)** 가 지배한다.
- 리소스 사용: `Grid(1), Block(32)` 최소 단위 실행.

### ❓ 왜 굳이 커널로 만드는가?
1) **동기화 회피**  
   Step 텐서가 GPU 메모리에 있을 때 CPU에서 계산하려면 D2H 복사가 필요하고, 이는 CUDA 스트림의 비동기 흐름을 끊는다.

2) **완전 Device-Side 실행**  
   학습 루프를 CUDA Graph Capture/Re-play로 고정하려면, 아주 작은 연산도 GPU 내부에서 끝내는 편이 유리하다.

---

## 4️⃣ 결론
- 기능 검증 완료: β decay 및 t=0 예외 처리가 정상 동작한다.
- 단독 커널로서는 런치 오버헤드가 지배적이므로, 추가 미세 최적화의 의미는 작다.
- 필요 시, AdamStep과의 **퓨전**을 통해 런치 비용 자체를 제거하는 방향을 고려할 수 있다.


다음 개선 아이디어(문서에 ‘Next Step’로 넣으면 폼 난다)

옵션 A: AdamStep 커널에 BiasCorr 로직을 퓨전해서 런치 1회를 제거

옵션 B: Step을 host에서 업데이트하더라도, 그래프 캡처 유지가 가능하도록 device-side step counter 설계