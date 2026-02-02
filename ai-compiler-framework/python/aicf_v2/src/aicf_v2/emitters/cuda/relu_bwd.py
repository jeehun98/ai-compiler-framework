from __future__ import annotations

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved


def relu_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    dy: int,
    y: int,
    out_dx: int,
    name: str = "relu_bwd",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    # typical contract: dx = dy * (y > 0)
    return emit_resolved(
        b,
        kind="relu_bwd",
        name=name,
        inputs=[dy, y],
        outputs=[out_dx],
        kind_id=ctx.ReluBwd,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
        constraints=constraints,
        hints=hints,
    )
