Stage6 Execution & Capture Architecture (AICF v2)
요약

Stage6는 AICF 실행 구조를 단순화하고 CUDA Graph Capture/Replay를 안정화하기 위한 단계다.

핵심 변화:

AICFBackend 제거

Python → C++ 단일 진입점: _C.op_call(kind, ins, outs, attrs)

실행 흐름을
IR → LoweredOps → BindingPlan → Static Env → Execute / Capture
로 명확히 분리

이전 gradient mismatch의 원인은 optimizer 구현 문제가 아니라
검증 시점(optimizer 이후/이전) 혼동이었음

1. 전체 실행 파이프라인 개요

Stage6 기준 실행 파이프라인은 아래 5단계로 구성된다.

IRGraph
  ↓ trace_ir
LoweredOps (op + vids + attrs)
  ↓ lower_to_backend_ops
BindingPlan (inputs / params / statics)
  ↓ build_binding_plan
Static Env (CUDA tensors, pointer-stable)
  ↓ executor.run
Execute / CUDA Graph Capture & Replay

핵심 설계 목표

출력/중간 버퍼를 Python에서 즉석 할당하지 않는다

모든 CUDA Tensor 포인터는 capture 전에 확정

실행 시 Python은 _C.op_call 호출만 담당

2. IR → LoweredOps
2.1 IRGraph (의미 단위 DAG)

IR에는 forward / backward / optimizer가 하나의 DAG로 포함된다.

예시 (train step 그래프):

Forward

Linear → ReLU → Save → Linear

Backward

MseGrad → LinearBwd → ReluBwd → LinearBwd

Optimizer

StepInc → BiasCorr → AdamStep(x4)

또는 SgdStep(x4)

IR의 특징:

의미 단위 연산 표현

실행 단위 아님

CUDA, 커널, 메모리 개념 없음

2.2 LoweredOps (실행 단위)

lower_to_backend_ops()는 IR node를 실행 가능한 primitive ops로 분해한다.

예시: Linear
Linear
  → gemm (x, W) -> y
  → bias_add (y, b) -> y   # in-place

Save
Save
  → copy_saved (x) -> saved

LinearBwd
LinearBwd
  → gemm (dY, W) -> dX
  → gemm (dY^T, X) -> dW
  → reduce_sum (dY) -> db

Optimizer lowering
StepInc   → step_inc
BiasCorr → bias_corr
AdamStep → adam_step
SgdStep  → sgd_step

LoweredOps의 성질

순차 리스트

각 op는 다음 형태:

{
  "op": "gemm",
  "inputs": [vid...],
  "outputs": [vid...],
  "attrs": {...},
  "kernel_id": Optional[str]
}


Python 실행 시 op 이름을 그대로 _C.op_call에 전달

3. Kernel Selection (StageB)

StageB는 LoweredOps에 대해 실제 실행할 커널(KID)을 결정한다.

3.1 역할

dtype / shape / alignment / layout / attrs 기반 KID 선택

vec2 / half2 업그레이드 적용

(선택) rewrite / fusion 수행 가능

3.2 현재 검증된 적용 사례

v2_kid_trace_by_id_test.py 기준:

bias_add_f16_vec2_v0

relu_f16_vec2_v0

relu_bwd_f16_vec2_v0

sgd_step_f16_half2_v0

LoweredOps와 runtime trace에서 동일 KID 호출 확인 → StageB 정상 동작.

4. BindingPlan & Static Env
4.1 BindingPlan의 역할

BindingPlan은
**“이 그래프를 실행하기 위해 필요한 모든 텐서의 계약서”**다.

구성

inputs

사용자 제공 (x, t, ...)

params

학습 파라미터 (W, b)

statics

forward intermediate

backward grads

optimizer state (m, v, step, bc_inv, ...)

모든 vid에 대해 확정되는 정보

shape

dtype

device

role (input / param / static)

4.2 Static Env (포인터 고정)

Executor는 BindingPlan을 기반으로:

statics 텐서를 미리 CUDA에 할당

이후 모든 실행에서 동일 포인터 재사용

이게 중요한 이유:

CUDA Graph capture는 포인터 주소 고정 필수

Python에서 매 step 새 텐서를 만들면 capture 불가

5. Executor.run (Eager 실행)
5.1 핵심 원칙

AICFBackend 없음

op_call_out 없음

실행은 오직 _C.op_call 단일 API

5.2 실행 흐름
env = ex.run(
    inputs={"x": x, "t": t},
    params=params,
    reuse_static=True,
)

내부 동작

inputs / params → plan의 vid에 bind

statics → 미리 할당된 텐서 재사용

lowered ops를 순서대로 실행

_C.op_call(kind, ins_tensors, outs_tensors, attrs)

특징

in-place op도 동일 경로

Python은:

스케줄링 ❌

메모리 관리 ❌

Python의 역할 = 계획 생성 + op_call 루프

6. CUDA Graph Capture / Replay
6.1 제공 API (C++)

graph_begin()

graph_end()

graph_launch()

graph_reset()

graph_stream()

6.2 Capture 흐름
graph_begin()
  - dedicated CUDA stream 전환
  - executor.run(...)
  - 모든 _C.op_call 기록
graph_end()
  - CUDA graph instance 확정
graph_launch()
  - replay
graph_reset()
  - graph 해제

6.3 현재 상태

eager / capture trace 완전히 동일

replay n회 후에도 결과 안정

capture 실패 시 원인은 대부분:

hidden allocation

포인터 변동

stream mismatch

7. 이전 Gradient Mismatch의 진짜 원인
7.1 증상
[check] dW0 maxdiff = 0.04
[check] d_lin0 maxdiff = 0.02


처음엔 gemm / optimizer / capture 문제로 의심됨.

7.2 실제 원인: 검증 시점 오류

Stage6 그래프는 기본적으로:

forward → backward → optimizer_step


를 한 번에 실행한다.

즉 run() 종료 시점에는:

gradients 계산 완료

weights는 이미 update됨

반면 PyTorch reference는 보통:

update 이전 weights 기준 backward

→ 서로 다른 W로 backward 수행
→ grad mismatch 발생

7.3 warmup=1에서 맞았던 이유

AICF_WARMUP=1:

step_inc no-op

adam_step lr=0 또는 skip

→ weights 변경 없음
→ PyTorch ref와 동일 chain
→ diff ≈ 0

8. 검증 전략 정리 (중요)
8.1 Gradient correctness 검증

아래 중 하나는 반드시 선택해야 한다.

warmup=1

optimizer 비활성화

weights snapshot

run 전 W.clone()

PyTorch ref backward는 clone 기준

그래프 분리

backward-only graph

optimizer-only graph

8.2 Training correctness 검증

Training에서는 grad 비교 ❌

대신 확인할 것:

W update 방향 / 크기

optimizer state (m/v/step)

loss 감소 추세

replay 안정성 (state drift)

9. Save / copy_saved 메모

현재:

Save → copy_saved


의미:

메모리 복사

bandwidth 소모

capture 노드 증가

향후 개선:

ReLU mask alias

activation 직접 참조

bitmask 저장

10. 운영 / 사용 가이드
10.1 기본 실행
ex = PlannedExecutor(ir, lowered, plan)
ex.run(inputs, params)

10.2 CUDA Graph Capture
_C.graph_begin()
ex.run(inputs, params)
_C.graph_end()

for _ in range(N):
    _C.graph_launch()

10.3 Warmup 모드

목적: grad correctness 검증

특징: optimizer 실질 동작 없음

사용 예:

AICF_WARMUP=1

10.4 Trace 켜기
ex.trace_enable(True)
ex.run(...)
trace = ex.trace_get()

11. 현재 구조의 핵심 정리

Python은 스케줄러가 아니다

Python은 메모리 관리자가 아니다

Python은:

IR 변환

lowering

plan 생성

_C.op_call 호출만 담당

이 구조의 결과:

CUDA Graph capture 안정

eager / replay 경로 동일

디버깅 포인트 명확

12. Roadmap

검증 스크립트 분리

check_grad_warmup.py

check_train_step.py

Save 최적화 (copy 제거)

replay stress test (1k~10k)

multi-step training graph 전략 비교

Appendix A. KID Trace Test

examples/python/python_framework_test/v2_kid_trace_by_id_test.py

목적:

StageB kernel selection이

lowered + runtime execution에 반영되는지 확인

통과 조건 예:

kid:sgd_step_f16_half2_v0