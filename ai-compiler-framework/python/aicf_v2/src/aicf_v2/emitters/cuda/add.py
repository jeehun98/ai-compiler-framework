from __future__ import annotations
from typing import Dict, Any

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

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
    )

def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,        # 최적화된 FWD Add EmitNode
    grad_y: int,          # dy Vid
    name: str = "add_bwd",
) -> Dict[int, int]:
    """
    최적화된 FWD add 노드를 바탕으로 BWD 연산을 누적합니다.
    y = a + c 이므로, da = dy, dc = dy 입니다. (Gradient Identity 전파)
    """
    a_vid = fwd_node.inputs[0]
    c_vid = fwd_node.inputs[1]

    # 단순히 grad_y(dy)를 각 입력의 미분값으로 전달합니다.
    # 만약 Shape 브로드캐스팅이 있었다면 여기서 ReduceSum 처리가 추가되어야 하나,
    # 현재 규격에서는 Identity 전파를 기본으로 합니다.
    return {
        a_vid: grad_y,
        c_vid: grad_y
    }