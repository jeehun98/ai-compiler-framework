from __future__ import annotations
import struct

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved


def sgd_step(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    P: int,
    G: int,
    outP: int,
    lr: float = 1e-3,
    name: str = "sgd_step",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    lr_f = float(lr)
    blob = struct.pack("<f", lr_f)

    return emit_resolved(
        b,
        kind="sgd_step",
        name=name,
        inputs=[P, G],
        outputs=[outP],
        kind_id=ctx.SgdStep,
        attr_schema=ctx.SCHEMA_SGDS,
        attr_blob=blob,
        attrs={"lr": lr_f},
        constraints=constraints,
        hints=hints,
    )
