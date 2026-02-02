from __future__ import annotations

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved


def add(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    a: int,
    c: int,
    out: int,
    name: str = "add",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    return emit_resolved(
        b,
        kind="add",
        name=name,
        inputs=[a, c],
        outputs=[out],
        kind_id=ctx.EltwiseAdd,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
        constraints=constraints,
        hints=hints,
    )
