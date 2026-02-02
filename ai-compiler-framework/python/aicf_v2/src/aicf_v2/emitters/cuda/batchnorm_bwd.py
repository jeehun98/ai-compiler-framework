from __future__ import annotations

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved


def batchnorm_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    dy: int,
    gamma: int,
    save_mean: int,
    save_rstd: int,
    out_dx: int,
    out_dgamma: int,
    out_dbeta: int,
    name: str = "batchnorm_bwd",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    return emit_resolved(
        b,
        kind="batchnorm_bwd",
        name=name,
        inputs=[x, dy, gamma, save_mean, save_rstd],
        outputs=[out_dx, out_dgamma, out_dbeta],
        kind_id=ctx.BatchNormBwd,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
        constraints=constraints,
        hints=hints,
    )
