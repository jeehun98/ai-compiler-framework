from __future__ import annotations
import struct
from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def bias_add(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    bias: int,
    out: int,
    broadcast_axis: int = -1,
    name: str = "bias_add",
) -> int:
    axis = int(broadcast_axis)
    blob = struct.pack("<q", axis)

    return emit_resolved(
        b,
        kind="bias_add",
        name=name,
        inputs=[x, bias],
        outputs=[out],
        kind_id=ctx.BiasAdd,
        attr_schema=ctx.SCHEMA_BADD,
        attr_blob=blob,
        attrs={"broadcast_axis": axis},
    )
