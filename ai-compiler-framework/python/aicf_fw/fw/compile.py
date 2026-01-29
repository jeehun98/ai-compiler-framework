# aicf_fw/fw/compile.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from aicf_fw.core_v2.backend_ops import BackendOp
from aicf_fw.core_v2.plan import build_binding_plan, apply_kernel_decisions_stageB
from aicf_fw.core_v2.rewrites.stageC_fuse_epilogue import stageC_fuse_gemm_epilogue
from aicf_fw.core_v2.exec import PlannedExecutor, ExecOptions
from aicf_fw.core_v2.op_attrs.registry import build_op_attr


@dataclass
class CompileArtifacts:
    ir: Any
    lowered: List[dict]
    plan: Any
    executor: PlannedExecutor


class TemplateCtx:
    """
    nn 레이어 템플릿이 최소한으로 요구하는 컨텍스트.
    - value id 발급
    - param name -> value id
    - (선택) value desc 저장 (shape/dtype/device)
    """
    def __init__(self, ir):
        self.ir = ir
        self._name_to_vid: Dict[str, int] = {}

    def bind_param_vid(self, name: str, vid: int) -> None:
        self._name_to_vid[name] = int(vid)

    def param_vid(self, name: str) -> int:
        return int(self._name_to_vid[name])

    def new_vid(self, name: str) -> int:
        # 이미 ir.values 구조가 있으니, 실제 구현은 ir에 value를 추가해야 함.
        # 여기선 "이미 만들어진 IR builder를 쓴다"는 전제로 id만 받는 형태로 스케치.
        return int(self.ir.new_value(name=name))


def compile_model_from_templates(
    *,
    model,
    optimizer=None,
    B: int,
    D: int,
    device: str,
    dtype,
    name: str = "aicf_fw_compile",
    warmup_runs: int = 0,
    warmup_inputs: Optional[Dict[str, Any]] = None,
    warmup_required: bool = False,
) -> CompileArtifacts:
    """
    fw-style compile의 핵심:
    (nn 템플릿 -> lowered ops) + (core_v2 pipeline) 결합
    """

    # 1) IR 생성 (여기서의 IR은 core_v2/ir.py 기반으로 가정)
    #    이미 fw가 trace_ir를 쓰고 있다면 그걸 재사용해도 됨.
    from aicf_fw.core_v2.trace import trace_ir  # 네 코드 구조에 맞춰 import 조정
    from aicf_fw.core_v2.ops import sym_tensor

    # build 함수에서 nn 템플릿을 호출해서 lowered(BackendOp)를 만들도록 구성
    backend_ops: List[BackendOp] = []

    def build():
        # inputs
        sx = sym_tensor(name="x", shape=(B, D), dtype=dtype, device=device)
        st = sym_tensor(name="t", shape=(B, D), dtype=dtype, device=device)

        # params 심볼 등록: model.named_parameters()를 기반으로 value id를 만들어놓고 ctx에 바인딩
        # (실제론 fw.module이 param 텐서를 가지고 있으니 name mapping이 이미 있을 것)
        # 여기선 "ir 상 param value를 sym_tensor로 만든다"로 스케치
        # Sequential prefix 기준 "0.W" 같은 이름을 그대로 유지
        # NOTE: 이 부분은 너 현재 fw/module.py의 파라미터 심볼링 방식에 맞춰 붙이면 됨.
        # ctx는 trace가 끝난 뒤 만들어야 하니, 여기서는 직접 sym_tensor로 가는게 더 단순함.

        # model이 Sequential이라고 가정하고, 내부 layers에 대해 템플릿 실행
        x_vid = sx.vid  # sym_tensor가 vid를 가진다고 가정(아니면 sx 자체가 value desc)
        for prefix, layer in model.iter_layers_with_prefix():  # 이건 네 sequential에 맞춰 제공하면 됨
            if hasattr(layer, "lower_template"):
                x_vid, ops = layer.lower_template(ctx=None, x_vid=x_vid, prefix=prefix)  # 아래에서 실제 ctx로 대체
                backend_ops.extend(ops)
            else:
                raise RuntimeError(f"layer {layer} has no lower_template")

        # loss / grad / optim 템플릿은 여기서 추가 가능
        # (mse_grad, linear_bwd, adam_step 등)
        # 지금은 구조만 잡기 위해 생략

    ir = trace_ir(build, name=name)

    # 2) 이제 ctx를 ir 기반으로 구성해서 param vid mapping을 채워야 함.
    #    위 build() 스케치에서 ctx=None로 했던 걸 실제로는 trace builder로 해결하는 방식이 더 깔끔함.
    #    "초미니 설계"니까: backend_ops를 ir 기반 lower_to_backend_ops로 만드는 방식과 공존시키자.
    #    => 아래는 현재 core_v2 lower 파이프라인을 태우는 '최소 변경 경로'

    from aicf_fw.core_v2.lower import lower_to_backend_ops
    lowered = lower_to_backend_ops(ir)  # StageA: IR에서 lowered dict 생성(현행 유지)

    # 3) StageB kernel decision
    lowered = apply_kernel_decisions_stageB(ir, lowered)

    # 4) StageC fuse
    lowered = stageC_fuse_gemm_epilogue(ir, lowered)

    # 5) OpAttrs 생성(선택): lowered에 oa를 붙이거나 dump용으로만 써도 됨
    value_descs = ir.values
    for i, lop in enumerate(lowered):
        lop_view = dict(lop)
        if "kind" not in lop_view and "op" in lop_view:
            lop_view["kind"] = lop_view["op"]
        oa = build_op_attr(lop_view, value_descs, op_id=i)
        # 필요하면 lowered[i]["op_attr_sig"]=oa.sig 같은 걸 추가 가능

    # 6) Plan
    plan = build_binding_plan(ir)

    # 7) Executor
    ex = PlannedExecutor(
        ir=ir,
        lowered=lowered,
        plan=plan,
        opts=ExecOptions(debug=False, require_kernel_id=True),
    )

    # 8) Warmup(옵션) - fw/module.py에서 이미 하는 방식이 있으면 거기 연결
    if warmup_runs > 0:
        if warmup_inputs is None and warmup_required:
            raise RuntimeError("warmup_required=True but warmup_inputs is None")
        if warmup_inputs is not None:
            for _ in range(warmup_runs):
                ex.run(inputs=warmup_inputs, params=dict(model.named_parameters()), reuse_static=True)

    return CompileArtifacts(ir=ir, lowered=lowered, plan=plan, executor=ex)
