# 📉 LayerNorm 커널 베이스라인 분석 (메모리 접근 패턴)

- **작성일:** 2026-02-16
- **디바이스:** NVIDIA RTX 3060
- **커널 타겟:** layernorm_fwd_f16_contig2d, layernorm_bwd_f16_contig2d

---

## 1️⃣ 벤치마크 결과

> **Status:** ⚠️ FWD 미흡 / ❌ BWD 치명적(스트라이드 접근)

| 모드 | 데이터 타입 | 입력 크기 (M×N) | 실행 시간 | 대역폭 (GB/s) | 상태 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FWD | F32 | 4096×4096 | 0.645 ms | 208.2 | ⚠️ |
| FWD | F16 | 4096×4096 | 0.285 ms | 235.7 | ⚠️ |
| BWD | F16 | 4096×4096 | 1.898 ms | 88.4 | ❌ Non-Coalesced |
| BWD | F16 | 32768×768 | 9.779 ms | 25.7 | ❌ Severe Strided |

---

## 2️⃣ 병목 원인 분석: Strided Memory Access

### ❌ Backward Pass 대역폭 붕괴 (최악: 25.7 GB/s)

- **커널 로직:** dgamma/dbeta 계산을 위해 Column(열) 단위로 모든 Row(행)을 순회하며 합산
- **데이터 레이아웃:** Row-major(가로 연속) 저장
- **문제:** 열 방향 접근은 연속성이 깨져, 많은 메모리 트랜잭션에서 **유효하게 쓰는 바이트가 극히 작아짐**
  - 직관적으로는 “수십~수백 바이트를 읽고 2바이트만 사용하는” 낭비가 반복됨
  - 이 낭비가 곧 유효 대역폭 붕괴로 이어짐
- **관찰:** N이 작아질수록(예: 768) 열 방향 stride가 더 불리해져 성능이 특히 악화됨

> 참고: 4096×4096에서는 88.4 GB/s로 덜 처참해 보일 수 있으나, baseline(≈300 GB/s) 대비 여전히 낮으며,
> 형상에 따라 낭비가 크게 달라질 수 있음을 보여준다.

---

### ⚠️ Forward Pass의 아쉬움 (235.7 GB/s)

- Copy baseline(≈300 GB/s) 대비 약 78% 수준
- 원인 후보:
  - `__half` 스칼라 로드/스토어로 인한 명령 발행 오버헤드
  - mean/var 계산을 위한 리덕션 단계에서의 동기화/명령 비용
- 방향:
  - `half2`/`float4` 기반 벡터 로드로 명령 발행 비용 감소
  - Welford/warp-level reduction 최적화로 리덕션 비용 축소

---

## 3️⃣ 향후 최적화 계획

### ✅ Step 1: BWD 리팩토링 (Coalesced Reduction)

- **전략 A:** 타일(예: 32×32)을 가로로 Coalesced Load → Shared/Warp Reduction으로 Column 합 누적
- **전략 B:** dgamma/dbeta 계산만을 위해 레이아웃 변환(부분 transpose) 또는 prepacked bias/scale 형태 고려

목표는 “열 방향 접근” 자체를 제거하거나, 타일링으로 **가로 접근 기반**으로 바꾸는 것이다.

---

### ✅ Step 2: FWD 벡터화

- `half2` 또는 `float4` 단위로 로드/스토어하여 스칼라 명령 발행 수를 감소
- 기대 목표: 280~300 GB/s 수준

---

## 4️⃣ 결론

현재 LayerNorm 구현은 정합성은 확보했지만,
Backward Pass에서 Row-major 데이터에 대해 Column-wise reduction을 수행하여
**Non-coalesced/strided 접근**이 발생했고 성능이 크게 붕괴했다.

이는 GPU에서 **Coalesced Access가 성능에 얼마나 절대적인지**를 보여주는 사례이며,
BWD의 접근 패턴을 타일 기반 Coalesced Reduction으로 바꾸는 것이 최우선 과제다.
