AICF Code Structure

이 문서는 AICF 프레임워크의 주요 디렉토리와 파일들의 역할을 설명합니다.
AICF는 크게 다음 네 계층으로 구성됩니다.

Graph Capture
↓
Compilation
↓
Emitter Resolution
↓
Runtime Execution
Top-Level Structure
aicf_v2
 ├─ builder
 ├─ graph
 ├─ compile
 ├─ emitters
 ├─ runtime
 └─ backends.cuda

각 계층은 다음 역할을 담당합니다.

Layer	역할
builder	연산 그래프 캡처
graph	Operator IR 정의
compile	IR 최적화 및 실행 계획
emitters	연산 → GPU kernel ABI 변환
runtime	GPU 실행 엔진
backends.cuda	CUDA 커널 바인딩
builder
builder.py
역할

Builder는 사용자 코드에서 호출된 연산들을 Operator Graph 형태로 캡처합니다.

예

y = relu(linear(x))

실제로 수행되는 일

builder.emit()
builder.emit()

Builder 내부에는 다음 정보가 저장됩니다.

ops
values
input_vids
output_vids
param_vids
state_vids
graph
graph.py
역할

Operator IR 정의.

핵심 클래스

Op

구성

kind
inputs
outputs
attrs
constraints
hints
kind_id
attr_schema
attr_blob

Op 객체는 AICF에서 실행 단위 역할을 합니다.

compile
compile/
 ├─ compile.py
 ├─ plan.py
 ├─ lower.py
 └─ passes/

컴파일 단계는 IR 최적화와 실행 정책 결정을 담당합니다.

compile.py
compile_cuda()

역할

IR 최적화
Execution plan 생성

실행 흐름

optimize_ir()
↓
make_exec_plan_cuda()
↓
CompiledProgram
plan.py
make_exec_plan_cuda()

역할

runtime execution plan 생성

생성 구조

ExecPlan
 ├─ ops
 └─ alias

alias는 in-place 실행 정책을 의미합니다.

lower.py
lower_ir_cuda()

역할

Op → LoweredOp 변환 (선택적)

현재 구조에서는 Op 자체가 실행 IR이므로
lower 단계는 디버그 / 검증 용도로 사용됩니다.

passes/
passes/
 ├─ pipeline.py
 └─ fusion.py

역할

graph optimization
pattern rewrite
fusion

패스는 semantic preserving transformation만 수행합니다.

emitters
emitters/cuda

Emitter는 연산 의미를 GPU kernel ABI로 해석하는 계층입니다.

base.py

공통 emitter 유틸리티.

핵심 함수

emit_resolved()

역할

Op 생성
kernel ABI 설정
bitmask fingerprint 기록
context.py

Emitter 실행 환경.

포함 내용

OpKind mapping
schema ID 정의
dynamic module loader

예

Gemm
Relu
LayerNorm
Softmax
AdamStep
개별 emitter

예

gemm.py
relu.py
softmax.py
layernorm.py
adam_step.py

각 emitter는 다음 역할을 수행합니다.

operator semantics 해석
kernel ABI 생성
attribute packing
runtime
runtime/
 ├─ cuda_exec.py
 ├─ alloc.py
 └─ graph_capture.py

Runtime은 GPU 실행 엔진입니다.

cuda_exec.py

핵심 실행기

CudaExecutor

역할

compiled program 실행
CUDA kernel launch
CUDA graph 실행

핵심 실행 루프

for op in plan.ops:
    op_call(...)
alloc.py

메모리 관리.

역할

vid → tensor slot binding
tensor allocation
graph_capture.py

CUDA Graph 실행 지원.

핵심 기능

capture_cuda_graph()
replay_cuda_graph()

그래프 캐시 키

(mode, plan_id, feed_signature)
backends.cuda
backends/cuda
 ├─ registry.py
 ├─ bridge.py
 └─ attrs.py

CUDA 커널과 Python 사이의 인터페이스 계층입니다.

registry.py

커널 메타데이터 관리.

예

OpKind mapping
schema verification
bridge.py

Python ↔ CUDA extension 연결.

핵심 함수

op_call()

역할

CUDA kernel launch
stream management
attrs.py

Attribute serialization 정의.

역할

kernel attribute packing
ABI consistency
Execution Flow (Code Perspective)

전체 실행 흐름

User Model
   ↓
Builder
   ↓
Operator Graph
   ↓
compile_cuda()
   ↓
ExecPlan
   ↓
CudaExecutor.run_compiled()
   ↓
Kernel Launch
Design Philosophy

AICF는 다음 설계 원칙을 따릅니다.

Semantic unit preservation

연산 의미 단위를 보존하면서 최적화를 수행합니다.

Emitter-driven execution

Emitter가 연산 의미를 kernel ABI로 해석합니다.

Lightweight runtime

Runtime은 GPU instruction interpreter 구조입니다.

Explicit kernel control

각 연산은 명확한 kernel ABI를 통해 실행됩니다.

Recommended Reading Order

AICF 코드를 이해하려면 다음 순서로 보는 것이 좋습니다.

1. graph.py
2. builder.py
3. emitters/base.py
4. compile/compile.py
5. compile/plan.py
6. runtime/cuda_exec.py