from __future__ import annotations
import struct
from typing import List

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved


def batchnorm_fwd(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    inputs: List[int],
    outputs: List[int],
    eps: float = 1e-5,
    use_running_stats: bool = False,
    name: str = "batchnorm_fwd",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    eps_f = float(eps)
    urs = 1 if bool(use_running_stats) else 0
    blob = struct.pack("<fI", eps_f, urs)

    return emit_resolved(
        b,
        kind="batchnorm_fwd",
        name=name,
        inputs=list(inputs),
        outputs=list(outputs),
        kind_id=ctx.BatchNormFwd,
        attr_schema=ctx.SCHEMA_BNEP,
        attr_blob=blob,
        attrs={"eps": eps_f, "use_running_stats": bool(use_running_stats)},
        constraints=constraints,
        hints=hints,
    )
