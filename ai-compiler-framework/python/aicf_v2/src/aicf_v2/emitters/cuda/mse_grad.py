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
    pred: int,
    target: int,
    out: int,
    scale: float | None = None,
    name: str = "mse_grad",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """MSE Gradient 연산을 IR에 기록합니다."""
    
    if scale is None:
        # 기본 경로: schema=0, empty blob (커널 내부 기본값 2/N 사용)
        return emit_resolved(
            b,
            kind="mse_grad",
            name=name,
            inputs=[pred, target],
            outputs=[out],
            kind_id=ctx.MseGrad,
            attr_schema=0,
            attr_blob=b"",
            attrs={},
            constraints=constraints,
            hints=hints,
        )

    # 스케일이 명시된 경우: SCHEMA_MSEG 사용
    sc = float(scale)
    blob = struct.pack("<f", sc)

    return emit_resolved(
        b,
        kind="mse_grad", # 최적화 패스에서 kind를 하나로 유지하는 것이 유리하므로 kind 통일 권장
        name=name,
        inputs=[pred, target],
        outputs=[out],
        kind_id=ctx.MseGrad,
        attr_schema=ctx.SCHEMA_MSEG,
        attr_blob=blob,
        attrs={"scale": sc},
        constraints=constraints,
        hints=hints,
    )

def emit_bwd(b: Builder, ctx: CudaEmitContext, fwd_node: Any, grad_y: int) -> Dict[int, int]:
    """MSE Grad는 대개 미분 그래프의 끝단이므로 역전파를 수행하지 않습니다."""
    return {}