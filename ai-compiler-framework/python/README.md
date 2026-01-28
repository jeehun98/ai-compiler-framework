AICF Framework (Minimal Core)

AICF는 PyTorch-like 사용자 API 위에
IR 기반 컴파일 → 커널 선택 → CUDA Graph 캡처/리플레이 파이프라인을 얹은
경량 AI 컴파일러 프레임워크입니다.

사용자는 일반적인 Python 프레임워크처럼 모델을 정의하고 학습하지만,
내부적으로는 연산 그래프를 IR로 변환하고, 커널 단위로 최적화된 실행 경로를 구성합니다.

핵심 목표

사용자 경험: PyTorch 스타일 API (nn / optim / model.compile)

컴파일러 구조: IR → Lower → Rewrite → Plan → Exec

런타임 최적화: kernel id 기반 실행, CUDA Graph capture / replay

검증 가능성: 실행된 kernel id를 trace로 확인 가능

전체 구조 개요
aicf_fw/
 ├─ nn/        # 모델 정의 (사용자 API)
 ├─ optim/     # 옵티마이저 (Adam 등)
 ├─ fw/        # compile / train_step / capture / replay
 └─ core_v2/   # IR 기반 컴파일러 & 런타임
      ├─ ir.py
      ├─ trace.py
      ├─ ops.py
      ├─ lower.py
      ├─ plan.py
      ├─ rewrites/
      │    └─ stageC_fuse_epilogue.py
      ├─ exec.py
      ├─ printer.py
      └─ op_attrs/
           ├─ registry.py
           ├─ gemm.py
           ├─ gemm_epilogue.py
           ├─ bias_add.py
           ├─ relu.py
           ├─ relu_bwd.py
           ├─ mse_grad.py
           ├─ reduce_sum.py
           ├─ copy.py
           └─ sgd_step.py

사용자 관점: PyTorch-like API
from aicf_fw.nn import Sequential, Linear, ReLU
from aicf_fw.optim import Adam

model = Sequential(
    Linear(D, D),
    ReLU(),
    Linear(D, D),
).to("cuda")

opt = Adam(model, lr=1e-3)

model.compile(
    optimizer=opt,
    B=B, D=D,
    device="cuda",
    dtype=torch.float32,
)

model.train_step({"x": x, "t": t})
model.capture({"x": x, "t": t})
model.replay(n=3, sync=True)


nn/, optim/, fw/는 일반 Python 프레임워크처럼 사용

내부에서는 컴파일된 실행 경로가 사용됨

내부 구조: 컴파일러 파이프라인
1️⃣ Trace (IR 생성)

위치: core_v2/trace.py, core_v2/ops.py

역할:

Python 코드에서 호출된 연산을 IRGraph로 기록

텐서(shape/dtype/device)와 연산 의존성 보존

Python ops
  ↓
IRGraph (values + nodes)

2️⃣ Lower (Stage A)

위치: core_v2/lower.py

역할:

고수준 IR 연산을 커널 단위 연산(gemm, bias_add, relu, …) 으로 분해

IRGraph
  ↓
LoweredOps (backend op list)

3️⃣ Kernel Decision (Stage B)

위치: core_v2/plan.py

역할:

dtype / shape 조건을 바탕으로 kernel id 선택

예: f16 + D % 2 == 0 → vec2 / half2 kernel

LoweredOps
  ↓
LoweredOps + kernel_id

4️⃣ Rewrite / Fuse (Stage C)

위치: core_v2/rewrites/stageC_fuse_epilogue.py

역할:

여러 연산을 하나의 fused op로 재구성

예: gemm + bias_add + relu → gemm_epilogue

gemm → bias_add → relu
  ↓
gemm_epilogue

5️⃣ OpAttrs (의미 기반 메타데이터)

위치: core_v2/op_attrs/

역할:

lowered op를 표준화된 의미 표현(OpAttr) 로 변환

커널 선택, 디버그, 룰 기반 처리의 공통 언어

LoweredOps
  ↓
OpAttr(op_kind, shapes, dtypes, layout, params, kid)

6️⃣ Plan & Exec

위치: core_v2/plan.py, core_v2/exec.py

역할:

메모리 바인딩 계획 수립

executor를 통해 실제 CUDA 커널 실행

실행된 kernel id를 trace로 수집 가능

📊 데이터 흐름 다이어그램 (Mermaid)
flowchart TD
    A[User Python Code\n(nn / optim / fw)] --> B[Trace\nIRGraph]
    B --> C[Lower\nStage A]
    C --> D[Kernel Decision\nStage B]
    D --> E[Rewrite / Fuse\nStage C]
    E --> F[OpAttrs\nSemantic Meta]
    F --> G[Binding Plan]
    G --> H[Executor]
    H --> I[CUDA Kernels\n(+ CUDA Graph)]
    I --> J[Runtime Trace\n(kernel_id)]

CUDA Graph Capture / Replay

model.capture():

컴파일된 executor를 CUDA Graph로 캡처

model.replay(n):

동일한 실행 경로를 반복 실행

optimizer state / meta tensor 변경 가능

➡️ 그래프 구조는 고정, 메타 파라미터는 동적

이 구조의 핵심 의미

nn / optim / fw

사용자 경험 계층

core_v2

컴파일러 + 런타임

op_attrs

커널 선택과 최적화를 위한 “의미 레이어”

AICF는
“Python 프레임워크처럼 쓰이지만,
실행은 커널 컴파일러처럼 동작한다”
를 목표로 설계되었습니다.

테스트

test_v2_kid_trace_by_id.py

kernel id 선택 / fuse / 실행 검증

test_fw_mvp_train_step.py

fw API, train_step, capture/replay, optimizer meta mutation 검증

마무리

이 레포는 기능을 최대한 줄인 상태에서도

컴파일러 구조

커널 단위 최적화

CUDA Graph 기반 재실행
을 모두 보여줄 수 있도록 구성된 Minimal but Complete 구조입니다.