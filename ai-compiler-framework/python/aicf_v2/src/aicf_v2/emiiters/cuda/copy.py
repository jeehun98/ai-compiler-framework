from __future__ import annotations
from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def copy(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    out: int,
    name: str = "copy",
) -> int:
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
    )
