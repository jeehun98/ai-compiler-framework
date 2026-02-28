from __future__ import annotations
import struct
from typing import Dict, Any

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved, OpFlags # OpFlags 추가
from . import reduce_sum

def emit(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    bias: int,
    out: int,
    broadcast_axis: int = -1,
    name: str = "bias_add",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """BiasAdd Forward 연산을 IR에 기록합니다."""
    axis = int(broadcast_axis)
    blob = struct.pack("<q", axis)

    # 정적 속성: BiasAdd는 Element-wise 연산입니다.
    # 인덱스 하드코딩 방지를 위한 role 추가
    in_role = ["x", "bias"]
    static = OpFlags.IS_ELEMENTWISE

    return emit_resolved(
        b,
        kind="bias_add",
        name=name,
        inputs=[x, bias],
        outputs=[out],
        kind_id=ctx.BiasAdd,
        attr_schema=ctx.SCHEMA_BADD,
        attr_blob=blob,
        attrs={
            "broadcast_axis": axis,
            "in_role": in_role,
        },
        constraints=constraints,
        hints=hints,
        static_flags=static, # 비트 주입
    )

def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,
    grad_y: int,
    name: str = "bias_add_bwd",
) -> Dict[int, int]:
    """dx = dy, db = reduce_sum(dy, axis=0)"""
    x_vid = fwd_node.inputs[0]
    bias_vid = fwd_node.inputs[1]
    
    grads = {x_vid: grad_y}

    bias_spec = b.values[bias_vid].spec
    db_vid = b.value(f"{name}.db", bias_spec)
    
    reduce_sum.emit(
        b,
        ctx,
        x=grad_y,
        out=db_vid,
        axis=0, 
        name=f"{name}.reduce_db"
    )
    
    grads[bias_vid] = db_vid
    return grads