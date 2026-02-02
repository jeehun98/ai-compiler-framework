from __future__ import annotations
from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def layernorm_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    gamma: int,
    dy: int,
    out_dx: int,
    out_dgamma: int,
    out_dbeta: int,
    name: str = "layernorm_bwd",
) -> int:
    return emit_resolved(
        b,
        kind="layernorm_bwd",
        name=name,
        inputs=[x, gamma, dy],
        outputs=[out_dx, out_dgamma, out_dbeta],
        kind_id=ctx.LayerNormBwd,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
    )
