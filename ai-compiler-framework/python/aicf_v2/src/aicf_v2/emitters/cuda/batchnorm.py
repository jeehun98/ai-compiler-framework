from __future__ import annotations
import struct
from typing import List, Dict, Any

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def emit(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    inputs: List[int],
    outputs: List[int],
    eps: float = 1e-5,
    use_running_stats: bool = False,
    name: str = "batchnorm",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """BatchNorm Forward 연산을 IR에 기록합니다."""
    eps_f = float(eps)
    urs = 1 if bool(use_running_stats) else 0
    # BNEP Schema: [eps(f32), flags(u32)]
    blob = struct.pack("<fI", eps_f, urs)

    return emit_resolved(
        b,
        kind="batchnorm",
        name=name,
        inputs=list(inputs),
        outputs=list(outputs),
        kind_id=ctx.BatchNormFwd,
        attr_schema=ctx.SCHEMA_BNEP,
        attr_blob=blob,
        attrs={"eps": eps_f, "use_running_stats": bool(use_running_stats)},
        constraints=constraints,
        hints=hints,
    )

def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,        # 최적화된 FWD BatchNorm EmitNode
    grad_y: int,          # dy Vid
    name: str = "batchnorm_bwd",
) -> Dict[int, int]:
    """
    최적화된 FWD batchnorm 노드를 역순회하며 BWD 연산을 누적합니다.
    Training 모드일 때 생성된 save_mean, save_rstd를 자동으로 찾아 바인딩합니다.
    """
    # Inference 모드였다면 미분 전파 중단 (또는 필요시 구현)
    if fwd_node.attrs.get("use_running_stats", False):
        return {}

    # 1. FWD 입력/출력 정보 추출
    # Training 인 경우 inputs: [x, gamma, beta], outputs: [y, save_mean, save_rstd]
    x = fwd_node.inputs[0]
    gamma = fwd_node.inputs[1]
    
    # 2. FWD 부산물(Saved Tensors) 추출
    save_mean = fwd_node.outputs[1]
    save_rstd = fwd_node.outputs[2]

    # 3. BWD 출력 Spec 정의
    x_spec = b.values[x].spec
    C = x_spec.shape[1]
    stat_spec = b.values[save_mean].spec # fp32[C]
    
    dx = b.value(f"{name}.dx", x_spec)
    dg = b.value(f"{name}.dgamma", stat_spec)
    db = b.value(f"{name}.dbeta", stat_spec)

    # 4. BWD Emit 호출
    emit_resolved(
        b,
        kind="batchnorm_bwd",
        name=name,
        inputs=[x, grad_y, gamma, save_mean, save_rstd],
        outputs=[dx, dg, db],
        kind_id=ctx.BatchNormBwd,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
    )

    # 5. grad_map 업데이트용 반환
    return {
        fwd_node.inputs[0]: dx, # d_x
        fwd_node.inputs[1]: dg, # d_gamma
        fwd_node.inputs[2]: db  # d_beta
    }