from __future__ import annotations
import struct
from typing import Any, Dict

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def emit(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    P: int,
    G: int,
    outP: int,
    lr: float = 1e-3,
    name: str = "sgd_step",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """SGD Optimizer의 가중치 업데이트 연산을 IR에 기록합니다."""
    lr_f = float(lr)
    # SGDS Schema: [lr(f32)]
    blob = struct.pack("<f", lr_f)

    return emit_resolved(
        b,
        kind="sgd_step",
        name=name,
        inputs=[P, G],
        outputs=[outP],
        kind_id=ctx.SgdStep,
        attr_schema=ctx.SCHEMA_SGDS,
        attr_blob=blob,
        attrs={"lr": lr_f},
        # 기본적으로 파라미터 버퍼에 Inplace 업데이트를 허용함
        constraints=constraints or {"inplace_ok": True},
        hints=hints,
    )

def emit_bwd(b: Builder, ctx: CudaEmitContext, fwd_node: Any, grad_y: int) -> Dict[int, int]:
    """SGD Step은 업데이트의 최종 단계이므로 역전파를 수행하지 않습니다."""
    return {}