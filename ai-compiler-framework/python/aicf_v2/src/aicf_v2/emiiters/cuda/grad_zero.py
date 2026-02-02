from __future__ import annotations
from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def grad_zero(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    name: str = "grad_zero",
) -> int:
    # out이 없고 inplace일 수도 있는데, 너 executor/plan 규칙에 맞춰서
    # outputs을 [x]로 둘지 별도의 out vid를 둘지 결정해.
    # 여기서는 "x를 out으로" 두는 형태(명시적).
    return emit_resolved(
        b,
        kind="grad_zero",
        name=name,
        inputs=[x],
        outputs=[x],
        kind_id=ctx.GradZero,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
        constraints={"inplace_ok": True},
    )
