from __future__ import annotations
import struct

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved


def adam_step(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    P: int,
    G: int,
    M: int,
    V: int,
    bc1: int,
    bc2: int,
    outP: int,
    outM: int,
    outV: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    name: str = "adam_step",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    lr_f = float(lr)
    b1 = float(beta1)
    b2 = float(beta2)
    eps_f = float(eps)
    blob = struct.pack("<ffff", lr_f, b1, b2, eps_f)

    # ABI: backend expects rank0 scalars for bc1/bc2; v2 uses (1,)
    abi_hints = {"view_rank0_inputs": [4, 5]}

    # merge optional hints
    merged_hints = dict(abi_hints)
    if hints:
        merged_hints.update(hints)

    return emit_resolved(
        b,
        kind="adam_step",
        name=name,
        inputs=[P, G, M, V, bc1, bc2],
        outputs=[outP, outM, outV],
        kind_id=ctx.AdamStep,
        attr_schema=ctx.SCHEMA_ADAM,
        attr_blob=blob,
        attrs={"lr": lr_f, "beta1": b1, "beta2": b2, "eps": eps_f},
        constraints=constraints or {"inplace_ok": True},
        hints=merged_hints,
    )
