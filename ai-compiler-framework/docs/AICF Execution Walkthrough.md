AICF Execution Walkthrough

이 문서는 AICF가 모델 정의 → 그래프 → 컴파일 → GPU 실행까지 어떻게 동작하는지 실제 코드 흐름을 따라 설명합니다.

예제로 다음 간단한 모델을 사용합니다.

Linear → ReLU → MSE Loss
1. Model Definition

사용자는 Python 레이어를 통해 모델을 정의합니다.

예시

y = linear(x, w, b)
y = relu(y)
loss = mse_loss(y, target)

이 단계에서 실제로 수행되는 일은 그래프 실행이 아니라 Operator Graph 캡처입니다.

2. Graph Capture (Builder)

모든 연산은 Builder를 통해 Operator Graph로 기록됩니다.

Builder
 └─ ops[]
      ├─ linear
      ├─ relu
      └─ mse_loss

각 연산은 Op 객체로 저장됩니다.

Op
 ├─ kind
 ├─ inputs
 ├─ outputs
 ├─ attrs
 ├─ constraints
 └─ hints

이 단계에서 실제 텐서 계산은 수행되지 않습니다.

3. Emitter Resolution

레이어가 생성될 때 Emitter가 호출됩니다.

Emitter는 연산의 의미를 GPU kernel ABI로 해석합니다.

핵심 함수

emit_resolved()

Emitter는 다음 정보를 Op에 기록합니다.

op.kind_id
op.attr_schema
op.attr_blob

즉

Operator semantics
      ↓
Kernel ABI

로 변환됩니다.

4. Operator Fingerprint

Emitter는 연산의 구조적 특징을 bitmask fingerprint로 기록합니다.

예

IS_GEMM_LIKE
IS_ELEMENTWISE
IS_REDUCE
IS_OPTIMIZER

이 정보는

graph optimization

fusion detection

kernel strategy

등에 사용됩니다.

5. Compile Phase

모델이 실행되면 compile_cuda()가 호출됩니다.

compile_cuda()

컴파일 단계는 실행 정책(runtime decisions) 만 결정합니다.

compile
 ├─ optimize_ir()
 └─ make_exec_plan_cuda()
6. IR Optimization
optimize_ir(builder)

역할

semantic-preserving graph transformations

예

operator fusion
dead op 제거
pattern rewrite

중요한 점은

연산의 의미 단위는 유지됩니다.

7. Execution Planning

다음 단계는 ExecPlan 생성입니다.

make_exec_plan_cuda()

생성되는 구조

ExecPlan
 ├─ ops
 └─ alias
Alias System

alias는 in-place 실행 정책을 의미합니다.

예

bias_add
sgd_step
step_inc

alias 예

out_vid → in_vid

runtime에서는

slots[out_vid] = slots[in_vid]

으로 적용됩니다.

8. Runtime Execution

실행은 CudaExecutor가 담당합니다.

CudaExecutor.run_compiled()

실행 흐름

bind_and_alloc_slots()
      ↓
apply alias
      ↓
execute ops
9. Kernel Launch

실제 GPU 실행은 다음 호출로 이루어집니다.

op_call(
    op.kind_id,
    inputs,
    outputs,
    op.attr_schema,
    op.attr_blob
)

즉 runtime은 instruction interpreter 구조입니다.

ExecPlan
   ↓
Kernel Launch Stream
10. Memory Model

AICF는 VID 기반 메모리 모델을 사용합니다.

vid → tensor slot

메모리 할당

bind_and_alloc_slots()

alias 적용

slot reuse
in-place execution
11. CUDA Graph Execution

반복 실행 시 CUDA Graph를 사용할 수 있습니다.

capture_cuda_graph()
replay_cuda_graph()

그래프 캐시 키

(mode, plan_id, feed_signature)

이를 통해

launch overhead 제거

가 가능합니다.

12. End-to-End Flow

전체 실행 흐름

Model Definition
      ↓
Builder Graph Capture
      ↓
Emitter Resolution
      ↓
Operator Graph
      ↓
compile_cuda()
      ↓
ExecPlan
      ↓
CudaExecutor
      ↓
Kernel Launch
Summary

AICF의 핵심 특징은 다음과 같습니다.

Operator = Execution IR

별도의 Lowered IR 없이 Op 자체가 실행 단위입니다.

Emitter-driven kernel resolution

Emitter가 연산 의미를 kernel ABI로 변환합니다.

Semantic-preserving optimization

최적화는 연산 의미 단위를 보존합니다.

Lightweight runtime

Runtime은 GPU instruction interpreter 구조입니다.

Why This Design?

이 구조는 다음 목표를 위해 설계되었습니다.

semantic clarity
minimal runtime overhead
explicit kernel control

즉

딥러닝 연산의 의미를 GPU 실행 계층까지 직접 연결하는 구조를 탐구합니다.