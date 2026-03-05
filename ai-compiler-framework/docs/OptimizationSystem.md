AICF Optimization System

이 문서는 AICF의 그래프 최적화 시스템을 설명합니다.

AICF는 딥러닝 연산을 의미 단위(Semantic Units) 로 취급하며,
최적화는 이 의미를 보존하는 범위 내에서 수행됩니다.

핵심 개념은 다음과 같습니다.

Semantic Unit Preservation
+ Operator Fingerprints
+ Pattern-based Optimization
Optimization Philosophy

대부분의 ML 컴파일러는 다음과 같은 방식으로 최적화를 수행합니다.

Pattern matching on operator strings

예

if op.kind == "gemm"

이 방식은 다음 문제가 있습니다.

문자열 비교 비용

패턴 확장 어려움

구조적 의미 부족

AICF 접근

AICF는 각 연산에 bitmask 기반 fingerprint를 부여합니다.

Op.static_flags
Op.derived_flags

이 fingerprint는 연산의 구조적 의미(Semantics) 를 표현합니다.

Operator Fingerprints

각 Operator는 OpFlags bitmask를 가집니다.

class OpFlags:

Flags는 크게 세 그룹으로 나뉩니다.

Static Flags
Trait Flags
Derived Flags
Static Flags (Semantics)

연산의 본질적인 성격을 나타냅니다.

예

IS_GEMM_LIKE
IS_ELEMENTWISE
IS_REDUCE
IS_OPTIMIZER
IS_NORM
IS_ACTIVATION

예시

Gemm → IS_GEMM_LIKE
Relu → IS_ELEMENTWISE | IS_ACTIVATION
ReduceSum → IS_REDUCE
AdamStep → IS_OPTIMIZER
Trait Flags (Operator Roles)

연산이 가지는 구조적 특성을 나타냅니다.

예

HAS_BIAS
HAS_STATE
INPLACE_PREF
TERMINAL

예시

bias_add → HAS_BIAS
adam_step → HAS_STATE
Derived Flags (Graph Context)

그래프 구조를 분석하여 패스 단계에서 계산되는 flag입니다.

예

SAFE_NODE
FUSION_BARRIER
DTYPE_F32
DTYPE_F16

예시

out_degree <= 1 → SAFE_NODE
Optimization Pipeline

그래프 최적화는 다음 단계로 수행됩니다.

Operator Graph
     ↓
Fingerprint Analysis
     ↓
Pattern Detection
     ↓
Graph Transformation
Fingerprint Queries

bitmask를 이용하면 빠르게 패턴을 탐지할 수 있습니다.

예

op.static_flags & IS_GEMM_LIKE

또는

(op.static_flags & QUERY_GEMM_ROOT) == QUERY_GEMM_ROOT

이 방식은 문자열 비교보다 훨씬 빠릅니다.

Example: Gemm Epilogue Fusion

다음 패턴을 고려합니다.

Gemm → BiasAdd → Relu

Fingerprint 기반 탐지

IS_GEMM_LIKE
↓
HAS_BIAS
↓
IS_ACTIVATION

이 패턴을 발견하면 다음 변환이 가능합니다.

GemmEpilogue

즉

3 operators
↓
1 fused kernel
Example: Optimizer Pattern

예

Grad → AdamStep

Fingerprint

IS_OPTIMIZER
HAS_STATE

이 패턴을 통해 다음 최적화를 수행할 수 있습니다.

state alias
inplace update
memory reuse
Fusion Safety

퓨전은 항상 안전하지 않습니다.

AICF는 다음 조건을 검사합니다.

SAFE_NODE
FUSION_BARRIER

예

out_degree > 1

이 경우 퓨전은 수행되지 않습니다.

Kernel Strategy Selection

Fingerprint는 커널 전략 선택에도 사용됩니다.

예

IS_GEMM_LIKE
+ DTYPE_F16

→ Tensor Core kernel

또는

IS_REDUCE

→ Warp reduction kernel

Optimization Example

다음 그래프가 있다고 가정합니다.

Linear → BiasAdd → Relu

Fingerprint 분석

IS_GEMM_LIKE
HAS_BIAS
IS_ACTIVATION

최적화 결과

GemmEpilogue kernel

그래프 변환

3 nodes → 1 node
Benefits

AICF optimization 시스템의 장점

Fast pattern detection

bitmask 연산은 매우 빠릅니다.

Semantic awareness

연산 의미를 직접 표현합니다.

Flexible rule definition

새로운 패턴을 쉽게 추가할 수 있습니다.

Kernel-driven optimization

커널 전략과 직접 연결됩니다.

Design Goals

AICF Optimization System은 다음 목표를 가집니다.

semantic preserving optimization
fast pattern detection
kernel-aware transformations

즉

연산 의미를 기반으로 최적화를 수행하면서 GPU 커널 전략과 직접 연결되는 구조를 제공합니다.

Summary

AICF Optimization System의 핵심은 다음입니다.

Operators carry semantic fingerprints
Optimization operates on fingerprints
Kernel strategies emerge from patterns

즉

Operator semantics
      ↓
Fingerprint
      ↓
Optimization
      ↓
Kernel strategy
Related Documents
Architecture.md
ExecutionWalkthrough.md
KernelSystem.md
CodeStructure.md