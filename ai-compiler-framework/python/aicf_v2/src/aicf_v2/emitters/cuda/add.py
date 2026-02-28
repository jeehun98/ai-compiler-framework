from __future__ import annotations
from typing import Dict, Any

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved, OpFlags # OpFlags 추가

def emit(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    a: int,
    c: int,
    out: int,
    name: str = "add",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """Element-wise Add Forward 연산을 IR에 기록합니다."""
    
    # 1. 정적 속성 선언: Add는 대표적인 Element-wise 연산입니다.
    static = OpFlags.IS_ELEMENTWISE
    
    # Inplace 가능 여부 확인 (예: a += c)
    if constraints and constraints.get("inplace_ok"):
        static |= OpFlags.INPLACE_PREF

    return emit_resolved(
        b,
        kind="add",
        name=name,
        inputs=[a, c],
        outputs=[out],
        kind_id=ctx.EltwiseAdd,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
        constraints=constraints,
        hints=hints,
        static_flags=static, # 본질 각인
    )

def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,
    grad_y: int,
    name: str = "add_bwd",
) -> Dict[int, int]:
    """
    최적화된 FWD add 노드를 바탕으로 BWD 연산을 누적합니다.
    """
    a_vid = fwd_node.inputs[0]
    c_vid = fwd_node.inputs[1]

    return {
        a_vid: grad_y,
        c_vid: grad_y
    }