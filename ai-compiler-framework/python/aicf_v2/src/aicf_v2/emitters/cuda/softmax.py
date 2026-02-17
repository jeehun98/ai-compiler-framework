# python/aicf_v2/src/aicf_v2/emitters/cuda/softmax.py
from __future__ import annotations
import struct

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def softmax(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    out: int,
    axis: int = -1,
    name: str = "softmax",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    # 'axis'를 소형 바이너리(int32)로 직렬화
    blob = struct.pack("<i", int(axis))

    return emit_resolved(
        b,
        kind="softmax",
        name=name,
        inputs=[x],
        outputs=[out],
        kind_id=ctx.Softmax,      # context에 정의된 Softmax ID 사용
        attr_schema=0,            # axis 하나만 있는 기본 스키마
        attr_blob=blob,
        attrs={"axis": int(axis)},
        constraints=constraints,
        hints=hints,
    )