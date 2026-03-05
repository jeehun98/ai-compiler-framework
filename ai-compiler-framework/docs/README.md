AICF — AI Compiler Framework

AICF (AI Compiler Framework) 는 딥러닝 연산의 수학적 의미(Semantics) 와 하드웨어 실행(Realization) 을 분리하기 위해 설계된 경량 AI 컴파일러입니다.

AICF는 연산을 단순한 실행 노드가 아니라 의미 단위(Semantic Unit) 로 다루며,
각 연산은 Emitter를 통해 GPU 커널 ABI로 해석됩니다.

이 구조를 통해 다음을 목표로 합니다.

Semantic-preserving optimization

Lightweight execution runtime

Kernel-level control over execution

Core Idea

대부분의 ML 컴파일러는 다음 구조를 사용합니다.

Operator IR
   ↓
Lowered IR
   ↓
Kernel Launch

AICF는 다른 접근을 사용합니다.

Operator (with kernel ABI metadata)
      ↓
Execution Plan
      ↓
Runtime Execution

즉,

Operator 객체 자체가 실행 IR 역할을 합니다.

Emitter 단계에서 이미 커널 ABI가 결정되기 때문에 별도의 Lowered IR이 필요하지 않습니다.

Architecture Overview
User Model
   │
   ▼
Builder (Graph Capture)
   │
   ▼
Operator Graph
   │
   ▼
Compile Phase
   ├─ IR Optimization
   └─ Execution Planning
          │
          ▼
       ExecPlan
          │
          ▼
Runtime Executor
          │
          ▼
CUDA Kernel Launch
Semantic Preservation

AICF의 최적화는 연산 의미 단위 보존(Semantic Unit Preservation) 을 전제로 합니다.

레이어 선언 시 각 연산에는 Emitter가 연결되며 다음 정보를 Op에 기록합니다.

kind_id
attr_schema
attr_blob

이는 GPU 커널 호출을 위한 ABI metadata입니다.

Emitter System

Emitter는 연산의 의미를 GPU 실행 형태로 해석하는 역할을 합니다.

핵심 함수

emit_resolved()

Emitter는 다음을 수행합니다.

Operator 생성
Kernel ID 결정
Attribute schema 결정
Attribute blob packing
Static flag 설정

즉

Operator semantics
      ↓
Kernel ABI

를 수행하는 단계입니다.

Operator Fingerprints

각 Op는 구조적 특징을 나타내는 bitmask fingerprint를 가집니다.

static_flags
derived_flags

예

IS_GEMM_LIKE
IS_ELEMENTWISE
IS_REDUCE
IS_OPTIMIZER

이 fingerprint는 다음에 사용됩니다.

graph optimization

fusion detection

kernel strategy selection

Compilation Pipeline

컴파일 단계는 실행 정책(Runtime decisions) 만 결정합니다.

compile_cuda()
   │
   ├─ optimize_ir()
   └─ make_exec_plan_cuda()
IR Optimization
optimize_ir(builder)

역할

semantic-preserving graph transformations

예

operator fusion

pattern rewrite

graph cleanup

Execution Planning
make_exec_plan_cuda(builder)

결정하는 것

inplace alias
runtime hints

예

out_vid → in_vid

이는 in-place execution을 의미합니다.

ExecPlan

ExecPlan은 runtime 실행 계획입니다.

@dataclass
class ExecPlan:
    ops: List[Op]
    alias: Dict[int, int]

구성

execution op stream
alias decisions

alias는 runtime에서

slot reuse
in-place execution

을 구현합니다.

Runtime Executor

Runtime은 매우 단순한 GPU instruction interpreter 구조입니다.

핵심 실행 루프

for op in plan.ops:
    op_call(
        op.kind_id,
        inputs,
        outputs,
        op.attr_schema,
        op.attr_blob
    )

ExecPlan은 GPU instruction stream과 유사한 역할을 합니다.

Memory Model

AICF는 VID 기반 메모리 모델을 사용합니다.

vid → tensor slot

메모리 할당

bind_and_alloc_slots()

In-place optimization

slots[out_vid] = slots[in_vid]
CUDA Graph Support

Runtime은 CUDA Graph 실행을 지원합니다.

capture_cuda_graph()
replay_cuda_graph()

그래프 캐시 키

(mode, plan_id, feed_signature)

이를 통해 반복 실행 시 launch overhead를 제거합니다.

Repository Structure
aicf_v2
 ├─ builder
 │    graph capture
 │
 ├─ graph
 │    operator IR definition
 │
 ├─ compile
 │    IR optimization
 │    execution planning
 │
 ├─ emitters
 │    operator → kernel ABI resolution
 │
 ├─ runtime
 │    execution engine
 │
 └─ backends.cuda
      CUDA kernel bindings
Key Characteristics

AICF는 기존 ML 컴파일러와 몇 가지 다른 특징을 가집니다.

Semantic-first design

연산 의미 단위를 기준으로 최적화를 수행합니다.

Emitter-first architecture

커널 ABI는 emitter 단계에서 결정됩니다.

Lightweight runtime

Runtime은 복잡한 scheduling 대신 instruction interpreter 구조를 사용합니다.

Explicit kernel control

각 연산은 명확한 kernel ABI를 통해 GPU 실행으로 연결됩니다.

Project Goal

AICF의 목표는 다음 질문을 탐구하는 것입니다.

딥러닝 연산의 수학적 의미를
GPU 실행 계층까지 명확하게 연결할 수 있는가?

이를 위해 AICF는 다음을 중심으로 설계되었습니다.

semantic-preserving compilation

emitter-driven kernel resolution

lightweight runtime execution

Project Status

Experimental AI compiler framework focusing on:

kernel-level execution control

semantic-preserving optimization

GPU runtime design

License

MIT License