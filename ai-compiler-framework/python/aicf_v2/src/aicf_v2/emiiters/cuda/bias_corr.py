from __future__ import annotations
import struct
from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def bias_corr(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    step: int,
    out_bc1_inv: int,
    out_bc2_inv: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    name: str = "bias_corr",
) -> int:
    b1 = float(beta1)
    b2 = float(beta2)
    blob = struct.pack("<ff", b1, b2)

    return emit_resolved(
        b,
        kind="bias_corr",
        name=name,
        inputs=[step],
        outputs=[out_bc1_inv, out_bc2_inv],
        kind_id=ctx.BiasCorr,
        attr_schema=ctx.SCHEMA_BCOR,
        attr_blob=blob,
        attrs={"beta1": b1, "beta2": b2},
    )
