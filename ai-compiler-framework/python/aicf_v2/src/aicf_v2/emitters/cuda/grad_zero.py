from __future__ import annotations

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved


def grad_zero(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    out: int,
    name: str = "grad_zero",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
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
        constraints=constraints or {"inplace_ok": True},
        hints=hints,
    )
