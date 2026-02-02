from __future__ import annotations
import struct
from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def batchnorm_fwd(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    gamma: int,
    beta: int,
    running_mean: int,
    running_var: int,
    out: int,
    eps: float = 1e-5,
    use_running_stats: bool = False,
    name: str = "batchnorm_fwd",
) -> int:
    eps_f = float(eps)
    urs = 1 if bool(use_running_stats) else 0
    blob = struct.pack("<fI", eps_f, urs)

    return emit_resolved(
        b,
        kind="batchnorm_fwd",
        name=name,
        inputs=[x, gamma, beta, running_mean, running_var],
        outputs=[out],
        kind_id=ctx.BatchNormFwd,
        attr_schema=ctx.SCHEMA_BNEP,
        attr_blob=blob,
        attrs={"eps": eps_f, "use_running_stats": bool(use_running_stats)},
    )
