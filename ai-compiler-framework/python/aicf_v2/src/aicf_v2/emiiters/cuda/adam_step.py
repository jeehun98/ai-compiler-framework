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
) -> int:
    lr_f = float(lr)
    b1 = float(beta1)
    b2 = float(beta2)
    eps_f = float(eps)

    blob = struct.pack("<ffff", lr_f, b1, b2, eps_f)

    # ABI 힌트(선택): bc1/bc2가 (1,)일 때 rank0 view 필요하면 여기에 남겨두고,
    # executor에서 generic하게 적용하도록 만들 수 있음.
    hints = {
        # "input_views": {4: "rank0", 5: "rank0"}
    }

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
        hints=hints,
    )
