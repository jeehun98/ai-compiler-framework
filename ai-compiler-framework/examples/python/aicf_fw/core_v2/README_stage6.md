Stage6 Execution & Capture Architecture (AICF v2)

요약

AICFBackend 제거

Python → C++ 단일 진입점 _C.op_call만 사용

IR → LoweredOps → BindingPlan → Env → Execute/Capture 로 흐름 단순화

이전 gradient mismatch의 원인은 optimizer가 아니라 검증 시점 오류

1. 전체 실행 파이프라인 개요

Stage6 기준 실행은 다음 5단계로 구성된다.

IRGraph
  ↓ trace_ir
LoweredOps (op + vids + attrs)
  ↓ lower_to_backend_ops
BindingPlan (inputs / params / statics)
  ↓ build_plan
Static Env (CUDA tensors, pointer-stable)
  ↓ executor.run
Execute / CUDA Graph Capture & Replay


이 구조의 핵심 목표는:

출력 버퍼를 Python에서 즉석 할당하지 않는다

모든 CUDA tensor 포인터는 capture 전에 확정

실행 시 Python 로직은 op_call 호출만 담당

2. IR → LoweredOps
2.1 IRGraph (예: v2_stage6_train1)

IR에는 forward / backward / optimizer가 하나의 DAG로 포함된다.

Forward: Linear → ReLU → Save → Linear

Backward: MseGrad → LinearBwd → ReluBwd → LinearBwd

Optimizer: StepInc → BiasCorr → AdamStep(x4)

IR은 의미 단위 연산이며, 실행 단위가 아니다.

2.2 LoweredOps (실행 단위)

lower_to_backend_ops()는 IR node를 실행 가능한 primitive ops로 분해한다.

예시:

Linear
  → gemm (x, W) -> y
  → bias_add (y, b) -> y

Save
  → copy_saved (x) -> saved

LinearBwd
  → gemm (dY, W) -> dX
  → gemm (dY^T, X) -> dW
  → reduce_sum (dY) -> db


Optimizer도 동일하게 lowering된다:

StepInc   → step_inc
BiasCorr → bias_corr
AdamStep → adam_step


중요한 점:

LoweredOps는 순차 리스트

각 op는 {op, inputs[vid], outputs[vid], attrs} 형태

Python 실행 시 op 이름을 그대로 _C.op_call에 전달

3. BindingPlan & Static Env
3.1 BindingPlan의 역할

BindingPlan은 **“이 그래프를 실행하기 위해 필요한 모든 텐서의 계약서”**다.

구성:

inputs: 사용자 제공 (x, t)

params: 학습 파라미터 (W, b)

statics:

forward intermediate

backward grads

optimizer state (m, v, step, bc_inv 등)

모든 vid에 대해 다음이 확정된다:

shape

dtype

device

role (input / param / static)

3.2 Static Env

Executor는 BindingPlan을 기반으로:

statics 텐서를 미리 CUDA에 할당

이후 실행에서는 항상 동일 포인터 재사용

이게 중요한 이유:

CUDA Graph capture는 포인터 주소가 고정되어야 한다

Python에서 매 step 새 텐서를 만들면 capture 불가능

4. Executor.run: eager 실행
4.1 핵심 원칙

AICFBackend 없음

op_call_out 없음

모든 실행은 _C.op_call 단일 API

4.2 실행 흐름
env = ex.run(
    inputs={"x": x, "t": t},
    params=params,
    reuse_static=True
)


내부 동작:

inputs / params를 plan의 vid에 bind

statics는 이미 할당된 텐서 사용

lowered ops를 순서대로 실행:

_C.op_call(kind, ins_tensors, outs_tensors, attrs)


in-place op (bias_add, adam_step)도 동일 경로

Python은 스케줄링도, 버퍼 관리도 안 함

5. CUDA Graph Capture / Replay
5.1 제공 API (C++)
graph_begin()
graph_end()
graph_launch()
graph_reset()
graph_stream()

5.2 Capture 흐름

graph_begin()

dedicated CUDA stream 생성

executor.run(...)

모든 _C.op_call이 capture stream에서 기록

graph_end()

CUDA graph 인스턴스 확정

graph_launch()

replay

graph_reset()

graph 해제

5.3 현재 상태

eager / capture trace가 완전히 동일

replay n회 후에도 결과 안정

capture 자체는 문제 없음

6. 이전 Gradient Mismatch의 진짜 원인
6.1 증상
[check] dW0 maxdiff = 0.04
[check] d_lin0 maxdiff = 0.02


→ 처음엔 gemm / optimizer / capture 문제로 의심됨

6.2 실제 원인

검증 시점 오류

Stage6 그래프는:

forward → backward → adam_step


를 한 번에 실행한다.

즉:

run()이 끝난 시점에는

gradients는 계산되었고

weights는 이미 adam_step으로 업데이트됨

그런데 PyTorch reference는:

“업데이트 이전 weights” 기준으로 grad를 계산

→ 서로 다른 W로 backward를 한 셈이 됨
→ grad mismatch 발생

6.3 왜 warmup=1에서는 맞았나?

AICF_WARMUP=1 모드에서는:

step_inc → no-op

adam_step → lr=0 또는 skip

weights가 안 바뀜

그래서:

AICF run 후에도 W 동일

PyTorch ref와 동일한 chain

diff ≈ 0

7. 검증 전략 정리 (중요)
7.1 Gradient correctness 검증

다음 중 하나를 반드시 택해야 한다:

warmup=1

optimizer 비활성화

backward correctness만 검증

weights snapshot

run 전에 W clone

ref backward는 clone 기준

optimizer 분리

backward-only graph

optimizer-only graph

7.2 Training correctness 검증

grad 비교 ❌

대신:

W update 방향/크기

m/v 상태 변화

loss 감소 추세

replay 안정성

8. Save / copy_saved에 대한 메모

현재:

Save → copy_saved


이는:

메모리 복사

bandwidth 소모

capture 시에도 추가 노드

향후 개선 여지:

ReLU mask를 alias로 저장

relu_bwd가 input activation 직접 참조

또는 bitmask 저장

9. 현재 구조의 핵심 정리

Python은 스케줄러가 아니다

Python은 메모리 관리자도 아니다

Python은:

IR 변환

lowering

plan 생성

_C.op_call 호출만 담당

이 구조 덕분에:

CUDA Graph capture가 안정됨

eager / replay 경로 동일

디버깅 포인트가 명확해짐

10. 다음 진행 추천 (Roadmap)

검증 스크립트 분리

check_grad_warmup.py

check_train_step.py

Save 최적화

copy 제거 or mask화

Replay stress test

replay 1k ~ 10k 반복

state drift 체크

Multi-step training 전략

step당 graph_launch vs

multi-step graph 캡처 비교