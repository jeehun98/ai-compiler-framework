from __future__ import annotations
import struct
from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def reduce_sum(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    out: int,
    axis: int = 0,
    name: str = "reduce_sum",
) -> int:
    ax = int(axis)
    blob = struct.pack("<q", ax)

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
    )
