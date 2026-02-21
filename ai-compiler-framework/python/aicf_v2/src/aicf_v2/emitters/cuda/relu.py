from __future__ import annotations
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
    name: str = "relu",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """ReLU Forward 연산을 IR에 기록합니다."""
    return emit_resolved(
        b,
        kind="relu",
        name=name,
        inputs=[x],
        outputs=[out],
        kind_id=ctx.EltwiseRelu,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
        constraints=constraints,
        hints=hints,
    )

def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,        # 최적화된 FWD ReLU EmitNode
    grad_y: int,          # dy Vid
    name: str = "relu_bwd",
) -> Dict[int, int]:
    """
    최적화된 FWD relu 노드를 바탕으로 BWD 연산을 누적합니다.
    dx = dy * (y > 0) 수식을 위해 FWD의 출력(y)을 입력으로 사용합니다.
    """
    # 1. FWD의 출력 Vid (y) 추출
    # fwd_node.outputs = [out_vid]
    y_vid = fwd_node.outputs[0]
    x_vid = fwd_node.inputs[0]

    # 2. 결과 Vid (dx) 생성
    x_spec = b.values[x_vid].spec
    dx_vid = b.value(f"{name}.dx", x_spec)

    # 3. ReLU BWD 전용 Emit 호출 (Mirroring)
    # y > 0 판정을 위해 fwd_node의 결과물인 y_vid를 참조함
    emit_resolved(
        b,
        kind="relu_bwd",
        name=name,
        inputs=[grad_y, y_vid],
        outputs=[dx_vid],
        kind_id=ctx.ReluBwd,
        attr_schema=0,
        attr_blob=b"",
        attrs={},
        constraints={"inplace_ok": True}, # dy 위치에 dx를 덮어쓰기 가능 (Lattice)
    )

    # 4. 상위 grad_map 갱신용 반환
    return {x_vid: dx_vid}