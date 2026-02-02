from __future__ import annotations
from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def relu_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    dy: int,
    out_dx: int,
    name: str = "relu_bwd",
) -> int:
    return emit_resolved(
        b,
        kind="relu_bwd",
        name=name,
        inputs=[x, dy],
        outputs=[out_dx],
        kind_id=ctx.ReluBwd,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
    )
