from __future__ import annotations
from typing import Dict, Any

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved, OpFlags # OpFlags 추가

def emit(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    out: int,
    name: str = "relu",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """ReLU Forward 연산을 IR에 기록합니다."""
    
    # 정적 속성 선언
    static = OpFlags.IS_ELEMENTWISE

    return emit_resolved(
        b,
        kind="relu",
        name=name,
        inputs=[x],
        outputs=[out],
        kind_id=ctx.EltwiseRelu,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
        constraints=constraints,
        hints=hints,
        static_flags=static, # 비트 주입
    )

def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,
    grad_y: int,
    name: str = "relu_bwd",
) -> Dict[int, int]:
    """dx = dy * (y > 0)"""
    y_vid = fwd_node.outputs[0]
    x_vid = fwd_node.inputs[0]

    x_spec = b.values[x_vid].spec
    dx_vid = b.value(f"{name}.dx", x_spec)

    # BWD에도 정적 속성 부여 (선택)
    bwd_static = OpFlags.IS_ELEMENTWISE

    emit_resolved(
        b,
        kind="relu_bwd",
        name=name,
        inputs=[grad_y, y_vid],
        outputs=[dx_vid],
        kind_id=ctx.ReluBwd,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
        constraints={"inplace_ok": True},
        static_flags=bwd_static,
    )

    return {x_vid: dx_vid}