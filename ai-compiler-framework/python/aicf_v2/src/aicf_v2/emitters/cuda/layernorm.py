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
    name: str = "layernorm",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """LayerNorm Forward 연산을 IR에 기록합니다."""
    eps_f = float(eps)
    # LNEP Schema: [eps(f32)]
    blob = struct.pack("<f", eps_f)

    return emit_resolved(
        b,
        kind="layernorm",
        name=name,
        inputs=list(inputs),
        outputs=list(outputs),
        kind_id=ctx.LayerNormFwd,
        attr_schema=ctx.SCHEMA_LNEP,
        attr_blob=blob,
        attrs={"eps": eps_f},
        constraints=constraints,
        hints=hints,
    )

def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,        # 최적화된 FWD LayerNorm EmitNode
    grad_y: int,          # dy Vid
    name: str = "layernorm_bwd",
) -> Dict[int, int]:
    """
    최적화된 FWD layernorm 노드를 바탕으로 BWD 연산을 누적합니다.
    FWD의 출력물인 [y, mean, rstd]를 BWD의 입력으로 자동 바인딩합니다.
    """
    # 1. FWD 입력 추출 (x, gamma, beta)
    x = fwd_node.inputs[0]
    gamma = fwd_node.inputs[1]
    
    # 2. FWD 출력(부산물) 추출
    # FWD emit 시 정의한 순서: [y, mean, rstd]
    # mean, rstd는 역전파 계산 시 수치적 안정성을 위해 필요함
    save_mean = fwd_node.outputs[1]
    save_rstd = fwd_node.outputs[2]

    # 3. BWD 출력 Spec 정의 (Lattice 정보 활용)
    x_spec = b.values[x].spec
    # LayerNorm은 보통 마지막 차원(Normalized Shape)에 대해 계산하므로 
    # dgamma, dbeta는 해당 차원의 크기를 가짐
    gamma_spec = b.values[gamma].spec
    
    dx = b.value(f"{name}.dx", x_spec)
    dg = b.value(f"{name}.dgamma", gamma_spec)
    db = b.value(f"{name}.dbeta", gamma_spec)

    # 4. BWD Emit 호출 및 누적
    emit_resolved(
        b,
        ctx,
        kind="layernorm_bwd",
        name=name,
        # LayerNormBwd 표준 입력: [dy, x, gamma, mean, rstd]
        inputs=[grad_y, x, gamma, save_mean, save_rstd],
        outputs=[dx, dg, db],
        kind_id=ctx.LayerNormBwd,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
    )

    # 5. grad_map 갱신을 위한 반환 (입력들에 대한 미분값 Vid 매핑)
    return {
        fwd_node.inputs[0]: dx, # d_x
        fwd_node.inputs[1]: dg, # d_gamma
        fwd_node.inputs[2]: db  # d_beta
    }