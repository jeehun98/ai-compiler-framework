from __future__ import annotations
import struct
from typing import List

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved


def layernorm_fwd(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    inputs: List[int],
    outputs: List[int],
    eps: float = 1e-5,
    name: str = "layernorm_fwd",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    eps_f = float(eps)
    blob = struct.pack("<f", eps_f)

    return emit_resolved(
        b,
        kind="layernorm_fwd",
        name=name,
        inputs=list(inputs),
        outputs=list(outputs),
        kind_id=ctx.LayerNormFwd,
        attr_schema=ctx.SCHEMA_LNEP,
        attr_blob=blob,
        attrs={"eps": eps_f},
        constraints=constraints,
        hints=hints,
    )
