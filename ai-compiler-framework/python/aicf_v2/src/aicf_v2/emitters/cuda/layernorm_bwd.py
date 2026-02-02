from __future__ import annotations
from typing import List

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved


def layernorm_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    inputs: List[int],
    outputs: List[int],
    name: str = "layernorm_bwd",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    return emit_resolved(
        b,
        kind="layernorm_bwd",
        name=name,
        inputs=list(inputs),
        outputs=list(outputs),
        kind_id=ctx.LayerNormBwd,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
        constraints=constraints,
        hints=hints,
    )
