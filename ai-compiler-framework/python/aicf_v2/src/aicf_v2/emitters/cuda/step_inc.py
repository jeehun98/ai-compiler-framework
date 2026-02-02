from __future__ import annotations

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved


def step_inc(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    step: int,
    out_step: int,
    name: str = "step_inc",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    return emit_resolved(
        b,
        kind="step_inc",
        name=name,
        inputs=[step],
        outputs=[out_step],
        kind_id=ctx.StepInc,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
        constraints=constraints or {"inplace_ok": True},
        hints=hints,
    )
