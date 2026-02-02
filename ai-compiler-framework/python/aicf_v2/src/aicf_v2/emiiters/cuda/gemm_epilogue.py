from __future__ import annotations
import struct
from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def gemm_epilogue(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    A: int,
    B: int,
    bias: int,
    out: int,
    transA: bool = False,
    transB: bool = False,
    relu: bool = True,
    name: str = "gemm_epilogue",
) -> int:
    ta = 1 if bool(transA) else 0
    tb = 1 if bool(transB) else 0
    r = 1 if bool(relu) else 0
    blob = struct.pack("<iii", ta, tb, r)

    return emit_resolved(
        b,
        kind="gemm_epilogue",
        name=name,
        inputs=[A, B, bias],
        outputs=[out],
        kind_id=ctx.GemmEpilogue,
        attr_schema=ctx.SCHEMA_GMEP,
        attr_blob=blob,
        attrs={"transA": bool(transA), "transB": bool(transB), "relu": bool(relu)},
    )
