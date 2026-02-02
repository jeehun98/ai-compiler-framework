from __future__ import annotations
import struct

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved


def mse_grad(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    pred: int,
    target: int,
    out: int,
    scale: float | None = None,
    name: str = "mse_grad",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    if scale is None:
        # default path: schema=0, empty blob
        return emit_resolved(
            b,
            kind="mse_grad",
            name=name,
            inputs=[pred, target],
            outputs=[out],
            kind_id=ctx.MseGrad,
            attr_schema=0,
            attr_blob=b"",
            attrs={},
            constraints=constraints,
            hints=hints,
        )

    sc = float(scale)
    blob = struct.pack("<f", sc)

    return emit_resolved(
        b,
        kind="mse_grad_scaled",
        name=name,
        inputs=[pred, target],
        outputs=[out],
        kind_id=ctx.MseGrad,
        attr_schema=ctx.SCHEMA_MSEG,
        attr_blob=blob,
        attrs={"scale": sc},
        constraints=constraints,
        hints=hints,
    )
