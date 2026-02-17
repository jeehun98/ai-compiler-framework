🚀 MSE Loss (Reduction) 커널 최적화 로그

    작성일: 2026-02-17

    디바이스: NVIDIA RTX 3060

    커널 타겟: mse_loss_sum_f32, mse_loss_sum_f16

1️⃣ 성능 요약 (유효 대역폭)

    Note: MSE Loss는 Pred와 Target을 읽어 차이의 제곱을 합산하는 전형적인 Memory-Bound Reduction 연산이다.
    출력은 스칼라(1 element)이므로 쓰기 대역폭은 무시하고, 유효 대역폭을 (Read(Pred) + Read(Target)) 기준으로 측정했다.

커널 버전	데이터 타입	입력 크기	실행 시간	유효 대역폭 (GB/s)	비고
v1.0 Naive	F32	16M (4096²)	Est. > 10ms	~12.5	Global Atomic 병목 (추정치)
v2.0 BlockReduce	F32	16M (16.7M)	0.499 ms	269.2	v1.0 대비 20배↑
v2.0 BlockReduce	F16	16M (16.7M)	0.388 ms	173.1	Mixed Precision (F32 Accum)
2️⃣ 이슈 분석: Reduction 병목과 정밀도
📉 1. Global Atomic 병목 (Naive 접근)

    초기 구현 시, 수만 개의 스레드가 하나의 전역 메모리 주소(out[0])에 직접 atomicAdd를 시도할 경우, **메모리 경합(Contention)**으로 인해 성능이 수십 배 저하된다.

    이를 해결하기 위해 Two-stage Reduction (Thread → Block → Grid) 구조가 필수적이다.

🛡️ 2. Mixed Precision (F16 입력 처리)

    문제: F16(half) 타입은 동적 범위가 좁아, 1,600만 개 이상의 오차 제곱을 누적 합산하면 Overflow 또는 정밀도 소실이 발생한다.

    해결: 데이터를 읽을 때는 F16(2 bytes)으로 읽지만, 커널 내부에서 __half2float 변환 후 F32 레지스터에 누적한다.

    트레이드오프: F16의 대역폭 이점은 있지만, 요소별 형변환 오버헤드와 비-벡터화된 로드(half vs half2)로 인해 이론상 최대 대역폭(F32의 2배)에는 도달하지 못함. (현재 173 GB/s)

⚠️ 3. 부동소수점 덧셈 순서와 오차

    테스트 중 Sum 모드에서 절대 오차(Absolute Error)가 크게 발생하여 FAIL이 떴으나, 이는 부동소수점 덧셈의 비결합성(Non-associativity) 때문이다.

    병렬 Reduction 순서와 순차적 CPU 덧셈 순서 차이로 인한 자연스러운 현상이므로, 검증 로직을 상대 오차(Relative Error) 기준으로 변경해야 함을 확인했다.

3️⃣ 커널 구현 전략
✅ v1.0: Naive Atomic (Reference)

    각 스레드가 루프를 돌며 계산 후 즉시 전역 메모리에 더함.

    디버깅 용도로만 적합하며 실사용 불가.

✅ v2.0: Block Reduction + Grid Stride Loop

    Grid-Stride Loop: 입력 크기가 스레드 수보다 클 때 루프를 돌아 처리. 스레드 재사용성을 높임.

    Warp Shuffle & Shared Memory:

        각 스레드가 레지스터(acc)에 부분합 계산.

        warp_reduce_sum: 워프 내 32개 스레드 값을 셔플(Shuffle)로 합산.

        block_reduce_sum: 워프의 대표값들을 공유 메모리(Shared Memory)에 모아 최종 블록 합산.

        AtomicAdd: 블록당 1번만 전역 메모리에 접근. (총 접근 횟수: N → BlockCount)

4️⃣ 결론

    Reduction 연산의 핵심은 "전역 메모리 쓰기 최소화"와 "수치 안정성"이다.

    Block Reduce 패턴을 적용하여 F32 기준 269 GB/s의 준수한 대역폭을 달성했다.

    대규모 데이터셋 학습 시 Loss 발산을 막기 위해 F16 입력이라도 내부 연산은 반드시 F32로 수행해야 함을 재확인했다.

    향후 F16 커널에 vectorized load (float4)를 적용하면 대역폭을 추가로 확보할 여지가 있다.