from __future__ import annotations
import struct
from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def gemm(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    A: int,
    B: int,
    C: int,
    out: int,
    transA: bool = False,
    transB: bool = False,
    name: str = "gemm",
) -> int:
    ta = 1 if bool(transA) else 0
    tb = 1 if bool(transB) else 0
    blob = struct.pack("<ii", ta, tb)

    return emit_resolved(
        b,
        kind="gemm",
        name=name,
        inputs=[A, B, C],
        outputs=[out],
        kind_id=ctx.Gemm,
        attr_schema=0,
        attr_blob=blob,
        attrs={"transA": bool(transA), "transB": bool(transB)},
    )
