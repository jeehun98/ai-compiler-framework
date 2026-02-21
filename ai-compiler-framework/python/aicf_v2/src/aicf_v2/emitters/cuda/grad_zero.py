from __future__ import annotations
from typing import Any, Dict

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def emit(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    out: int,
    name: str = "grad_zero",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """그래디언트 버퍼를 0으로 초기화하는 연산을 IR에 기록합니다."""
    return emit_resolved(
        b,
        kind="grad_zero",
        name=name,
        inputs=[x],
        outputs=[out],
        kind_id=ctx.GradZero,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
        # 기본적으로 Inplace를 허용하여 메모리를 절약합니다.
        constraints=constraints or {"inplace_ok": True},
        hints=hints,
    )

def emit_bwd(b: Builder, ctx: CudaEmitContext, fwd_node: Any, grad_y: int) -> Dict[int, int]:
    """GradZero는 초기화 연산이므로 역전파를 수행하지 않습니다."""
    return {}