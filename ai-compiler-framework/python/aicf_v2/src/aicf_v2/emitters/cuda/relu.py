from __future__ import annotations

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved


def relu(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    out: int,
    name: str = "relu",
    saved: list[int] | None = None,
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    return emit_resolved(
        b,
        kind="relu",
        name=name,
        inputs=[x],
        outputs=[out],
        kind_id=ctx.EltwiseRelu,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
        saved=saved,
        constraints=constraints,
        hints=hints,
    )
