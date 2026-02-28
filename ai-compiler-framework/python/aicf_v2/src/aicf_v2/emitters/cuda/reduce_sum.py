from __future__ import annotations
import struct
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
    axis: int = 0,
    name: str = "reduce_sum",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """합산(Reduction) Forward 연산을 IR에 기록합니다."""
    ax = int(axis)
    blob = struct.pack("<q", ax)

    # 정적 속성: Reduction 연산임을 명시
    static = OpFlags.IS_REDUCE

    return emit_resolved(
        b,
        kind="reduce_sum",
        name=name,
        inputs=[x],
        outputs=[out],
        kind_id=ctx.ReduceSum,
        attr_schema=ctx.SCHEMA_RSUM,
        attr_blob=blob,
        attrs={"axis": ax},
        constraints=constraints,
        hints=hints,
        static_flags=static, # 비트 주입
    )

def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,
    grad_y: int,
    name: str = "reduce_sum_bwd",
) -> Dict[int, int]:
    """dx = broadcast(grad_y)"""
    x_vid = fwd_node.inputs[0]
    axis = fwd_node.attrs["axis"]
    
    x_spec = b.values[x_vid].spec
    dx_vid = b.value(f"{name}.dx", x_spec)

    emit_resolved(
        b,
        kind="broadcast",
        name=name,
        inputs=[grad_y],
        outputs=[dx_vid],
        kind_id=ctx.Copy,
        attr_schema=0,
        attr_blob=b"",
        attrs={"axis": axis, "target_shape": x_spec.shape},
        static_flags=OpFlags.IS_ELEMENTWISE, # 브로드캐스트는 성격상 elementwise에 가까움
    )

    return {x_vid: dx_vid}