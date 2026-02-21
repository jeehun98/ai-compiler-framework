from __future__ import annotations
import struct
from typing import Dict, Any

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def emit(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    out: int,
    axis: int = 0,
    name: str = "reduce_sum",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """합산(Reduction) Forward 연산을 IR에 기록합니다."""
    ax = int(axis)
    # RSUM Schema: [axis(i64)]
    blob = struct.pack("<q", ax)

    return emit_resolved(
        b,
        kind="reduce_sum",
        name=name,
        inputs=[x],
        outputs=[out],
        kind_id=ctx.ReduceSum,
        attr_schema=ctx.SCHEMA_RSUM,
        attr_blob=blob,
        attrs={"axis": ax},
        constraints=constraints,
        hints=hints,
    )

def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,        # 최적화된 FWD ReduceSum EmitNode
    grad_y: int,          # dy Vid (Reduced shape)
    name: str = "reduce_sum_bwd",
) -> Dict[int, int]:
    """
    최적화된 FWD reduce_sum 노드를 바탕으로 BWD 경로를 생성합니다.
    합산의 역전파는 Broadcast입니다. (dx = grad_y를 x의 형상으로 확장)
    """
    x_vid = fwd_node.inputs[0]
    axis = fwd_node.attrs["axis"]
    
    x_spec = b.values[x_vid].spec
    dx_vid = b.value(f"{name}.dx", x_spec)

    # 역전파를 위해 브로드캐스트(Copy/Expand) 연산 모듈을 호출합니다.
    # 여기서는 시스템의 단순성을 위해 전용 'broadcast' kind를 사용한다고 가정합니다.
    emit_resolved(
        b,
        kind="broadcast",
        name=name,
        inputs=[grad_y],
        outputs=[dx_vid],
        kind_id=ctx.Copy, # 백엔드 커널이 브로드캐스트를 지원하는 Copy라고 가정
        attr_schema=0,
        attr_blob=b"",
        attrs={"axis": axis, "target_shape": x_spec.shape},
    )

    return {x_vid: dx_vid}