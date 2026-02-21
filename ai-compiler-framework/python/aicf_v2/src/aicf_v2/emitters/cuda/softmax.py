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
    axis: int = -1,
    name: str = "softmax",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """Softmax Forward 연산을 IR에 기록합니다."""
    ax = int(axis)
    # SCHEMA 0 (기본): [axis(i32)]
    blob = struct.pack("<i", ax)

    return emit_resolved(
        b,
        kind="softmax",
        name=name,
        inputs=[x],
        outputs=[out],
        kind_id=ctx.Softmax,
        attr_schema=0,
        attr_blob=blob,
        attrs={"axis": ax},
        constraints=constraints,
        hints=hints,
    )

def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,        # 최적화된 FWD Softmax EmitNode
    grad_y: int,          # dy Vid
    name: str = "softmax_bwd",
) -> Dict[int, int]:
    """
    최적화된 FWD softmax 노드를 바탕으로 BWD 연산을 누적합니다.
    수식: dx = y * (dy - sum(y * dy))
    Mirroring을 통해 FWD의 출력(y)을 BWD의 입력으로 자동 사용합니다.
    """
    # 1. FWD 정보 추출
    x_vid = fwd_node.inputs[0]
    y_vid = fwd_node.outputs[0] # Softmax의 출력 y
    axis = fwd_node.attrs["axis"]

    # 2. 출력 Spec 정의 (x와 동일 형상)
    x_spec = b.values[x_vid].spec
    dx_vid = b.value(f"{name}.dx", x_spec)

    # 3. Softmax BWD 전용 Emit 호출
    # 백엔드 커널 규약: inputs=[y, dy], outputs=[dx]
    blob = struct.pack("<i", int(axis))
    emit_resolved(
        b,
        kind="softmax_bwd",
        name=name,
        inputs=[y_vid, grad_y],
        outputs=[dx_vid],
        kind_id=ctx.SoftmaxBwd,
        attr_schema=0,
        attr_blob=blob,
        attrs={"axis": axis},
    )

    # 4. 입력(x)에 대한 미분값 전파
    return {x_vid: dx_vid}