# 📉 ReduceSum (마지막 차원 유지) 커널 베이스라인 분석

- **작성일:** 2026-02-16
- **디바이스:** NVIDIA RTX 3060
- **커널 타겟:** reduce_sum_keep_lastdim_f32_to_f32, reduce_sum_keep_lastdim_f16_to_f16

---

## 1️⃣ 벤치마크 결과

> **Status:** ❌ 심각한 성능 붕괴 (병렬성/접근 패턴 불일치)

| 케이스 | 입력 크기 (M×N) | 실행 시간 | 대역폭 (GB/s) | 상태 | 주요 원인 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| F32 Global Sum | 16M×1 | 5.46 ms | 12.3 | ❌ | 병렬성 붕괴(블록 1개 수준) |
| F32 Bias Grad | 16K×1024 | 2.27 ms | 29.6 | ❌ | Column 접근으로 인한 Strided/Non-coalesced |
| F16 Global Sum | 16M×1 | 20.07 ms | 1.7 | 💀 | 병렬성 붕괴 + half 스칼라 경로 비효율 |
| F16 Bias Grad | 16K×1024 | 0.33 ms | 102.1 | ⚠️ | half2 벡터화로 일부 회복 |

---

## 2️⃣ 병목 원인 분석

### 💀 Global Sum (N=1): 병렬성 붕괴

- 현재 커널은 `grid(N, 1, 1)` 형태로 실행된다.
- **N=1이면 동시에 실행 가능한 블록이 사실상 1개**가 되어, GPU의 대부분 SM이 유휴 상태가 된다.
- 결과적으로 메모리/연산 자원을 거의 쓰지 못하고 성능이 급락한다.

---

### ❌ Bias Grad (N=1024): Column-wise 접근에 의한 Strided Access

- Row-major 레이아웃에서 Column 방향으로 읽으면 주소가 `stride=N` 만큼 점프한다.
- warp 내 스레드들이 연속 주소를 접근하지 못해 코얼레싱이 깨지고,
  많은 트랜잭션에서 유효 바이트가 작아져 DRAM/L2 효율이 급락한다.
- 그 결과 대역폭이 30 GB/s 수준으로 붕괴한다.

---

## 3️⃣ 해결 방향 (To-Do)

### ✅ 전략 1: Global Sum 전용 병렬 리덕션

- N=1 케이스는 “마지막 차원 유지 리덕션”이 아니라 사실상 **전체 합(스칼라)** 문제로 취급해야 한다.
- `grid(많은 블록)`으로 부분합을 만든 뒤,
  2단계 리덕션(또는 atomic)으로 최종 합을 만든다.
- 목표: Copy/Add baseline 대비 높은 비율(수십~수백 GB/s)을 확보

---

### ✅ 전략 2: Bias Grad용 타일 기반 Coalesced Reduction

- 읽기는 Row 방향으로 coalesced하게 수행하고,
  타일(예: 32×32)을 shared memory에 적재한 뒤
  column 합을 shared/warp reduction으로 누적한다.
- 핵심: **“읽기는 가로(Coalesced), 더하기는 세로(타일 내부)”**

- 목표: baseline 대비 큰 폭의 효율 개선(수백 GB/s 구간 진입)

---

## 4️⃣ 결론

ReduceSum은 “올바른 수학”이라도 **GPU 하드웨어에 맞는 병렬화/메모리 접근**이 없으면
성능이 붕괴한다는 것을 보여준다.

- N=1에서는 병렬성이 붕괴하므로 전용 병렬 리덕션이 필요하고,
- N이 큰 bias grad 형태에서는 column 접근을 제거하기 위한 타일 기반 reduction이 필수다.
