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
    step: int,
    out_bc1_inv: int,
    out_bc2_inv: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    name: str = "bias_corr",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """Bias Correction (Adam) 연산을 IR에 기록합니다."""
    b1 = float(beta1)
    b2 = float(beta2)
    # BCOR Schema: [beta1(f32), beta2(f32)]
    blob = struct.pack("<ff", b1, b2)

    return emit_resolved(
        b,
        kind="bias_corr",
        name=name,
        inputs=[step],
        outputs=[out_bc1_inv, out_bc2_inv],
        kind_id=ctx.BiasCorr,
        attr_schema=ctx.SCHEMA_BCOR,
        attr_blob=blob,
        attrs={"beta1": b1, "beta2": b2},
        constraints=constraints,
        hints=hints,
    )

def emit_bwd(b: Builder, ctx: CudaEmitContext, fwd_node: Any, grad_y: int) -> Dict[int, int]:
    """Bias Correction은 최적화 단계의 부산물이므로 역전파를 수행하지 않습니다."""
    return {}