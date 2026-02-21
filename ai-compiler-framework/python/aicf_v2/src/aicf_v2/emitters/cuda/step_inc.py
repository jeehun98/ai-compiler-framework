from __future__ import annotations
from typing import Any, Dict

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def emit(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    step: int,
    out_step: int,
    name: str = "step_inc",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """학습 스텝 카운트를 1 증가시키는 연산을 IR에 기록합니다."""
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
        # 카운터 변수이므로 인플레이스 업데이트가 효율적입니다.
        constraints=constraints or {"inplace_ok": True},
        hints=hints,
    )

def emit_bwd(b: Builder, ctx: CudaEmitContext, fwd_node: Any, grad_y: int) -> Dict[int, int]:
    """Step Increment는 제어용 연산이므로 역전파를 수행하지 않습니다."""
    return {}