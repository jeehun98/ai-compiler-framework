from __future__ import annotations
import struct
from typing import Dict, Any

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved
from . import reduce_sum  # d_bias 계산을 위해 reduce_sum 모듈 참조

def emit(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    bias: int,
    out: int,
    broadcast_axis: int = -1,
    name: str = "bias_add",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """BiasAdd Forward 연산을 IR에 기록합니다."""
    axis = int(broadcast_axis)
    # BADD Schema: [broadcast_axis(i64)]
    blob = struct.pack("<q", axis)

    return emit_resolved(
        b,
        kind="bias_add",
        name=name,
        inputs=[x, bias],
        outputs=[out],
        kind_id=ctx.BiasAdd,
        attr_schema=ctx.SCHEMA_BADD,
        attr_blob=blob,
        attrs={"broadcast_axis": axis},
        constraints=constraints,
        hints=hints,
    )

def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,        # 최적화된 FWD BiasAdd EmitNode
    grad_y: int,          # dy Vid
    name: str = "bias_add_bwd",
) -> Dict[int, int]:
    """
    최적화된 FWD bias_add 노드를 바탕으로 BWD 연산을 누적합니다.
    dx = dy (Identity)
    db = reduce_sum(dy, axis=...)
    """
    x_vid = fwd_node.inputs[0]
    bias_vid = fwd_node.inputs[1]
    
    # 1. FWD 속성에서 어느 축이 bias 축이었는지 확인
    axis = fwd_node.attrs["broadcast_axis"]
    
    # 2. dx 생성 (Identity 전파)
    # 별도의 커널 없이 grad_map에 직접 연결
    grads = {x_vid: grad_y}

    # 3. db 생성 (Bias에 대한 Gradient Reduction)
    bias_spec = b.values[bias_vid].spec
    db_vid = b.value(f"{name}.db", bias_spec)
    
    # [핵심 수정] keepdims 제거 및 reduce_sum 모듈의 실제 파라미터 규격 준수
    # reduce_sum.emit(b, ctx, x=..., out=..., axis=...)
    reduce_sum.emit(
        b,
        ctx,
        x=grad_y,
        out=db_vid,
        axis=0,  # bias_add_bwd 커널 혹은 reduce_sum이 이 축을 기준으로 동작
        name=f"{name}.reduce_db"
    )
    
    grads[bias_vid] = db_vid

    return grads