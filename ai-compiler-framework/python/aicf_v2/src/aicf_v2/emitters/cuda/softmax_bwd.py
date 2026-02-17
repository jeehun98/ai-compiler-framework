# python/aicf_v2/src/aicf_v2/emitters/cuda/softmax_bwd.py
from __future__ import annotations
import struct

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def softmax_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    out: int,       # Forward의 출력값 (softmax 결과)
    grad_out: int,   # Loss로부터 전파된 gradient
    grad_in: int,    # 계산될 입력에 대한 gradient
    axis: int = -1,
    name: str = "softmax_bwd",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    blob = struct.pack("<i", int(axis))

    return emit_resolved(
        b,
        kind="softmax_bwd",
        name=name,
        inputs=[out, grad_out],
        outputs=[grad_in],
        kind_id=ctx.SoftmaxBwd,
        attr_schema=0,
        attr_blob=blob,
        attrs={"axis": int(axis)},
        constraints=constraints,
        hints=hints,
    )