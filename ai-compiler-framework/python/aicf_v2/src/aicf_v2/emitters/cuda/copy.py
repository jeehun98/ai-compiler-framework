from __future__ import annotations
from typing import Dict, Any

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def emit(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    out: int,
    name: str = "copy",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """Tensor 복사(Identity) Forward 연산을 IR에 기록합니다."""
    return emit_resolved(
        b,
        kind="copy",
        name=name,
        inputs=[x],
        outputs=[out],
        kind_id=ctx.Copy,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
        constraints=constraints,
        hints=hints,
    )

def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,        # 최적화된 FWD Copy EmitNode
    grad_y: int,          # dy Vid
    name: str = "copy_bwd",
) -> Dict[int, int]:
    """
    최적화된 FWD copy 노드를 바탕으로 BWD 경로를 연결합니다.
    y = x 이므로, dx = dy 입니다. 
    별도의 복사 노드 생성 없이 grad_y를 그대로 전파하여 최적화합니다.
    """
    x_vid = fwd_node.inputs[0]
    
    # Identity 전파
    return {x_vid: grad_y}