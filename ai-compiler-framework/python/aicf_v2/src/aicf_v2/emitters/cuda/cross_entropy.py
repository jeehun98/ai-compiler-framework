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
    logits: int,
    targets: int,
    out: int,
    ignore_index: int = -100,
    reduction: int = 0, # 0: mean, 1: sum, 2: none
    name: str = "cross_entropy",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """Cross Entropy Loss Forward 연산을 IR에 기록합니다."""
    ii = int(ignore_index)
    red = int(reduction)
    # XENT Schema: [ignore_index(i32), reduction(i32)]
    blob = struct.pack("<ii", ii, red)

    return emit_resolved(
        b,
        kind="cross_entropy",
        name=name,
        inputs=[logits, targets],
        outputs=[out],
        kind_id=ctx.CrossEntropyFwd,
        attr_schema=ctx.SCHEMA_XENT,
        attr_blob=blob,
        attrs={"ignore_index": ii, "reduction": red},
        constraints=constraints,
        hints=hints,
    )

def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,        # 최적화된 FWD CrossEntropy EmitNode
    grad_y: int,          # dy (grad_out) Vid
    name: str = "cross_entropy_bwd",
) -> Dict[int, int]:
    """
    최적화된 FWD cross_entropy 노드를 바탕으로 BWD 연산을 누적합니다.
    [수정] emit_resolved 호출 시 키워드 인자 규격을 엄격히 준수합니다.
    """
    # 1. FWD 입력 및 속성 추출 (Mirroring)
    logits_vid = fwd_node.inputs[0]
    targets_vid = fwd_node.inputs[1]
    
    ii = fwd_node.attrs["ignore_index"]
    red = fwd_node.attrs["reduction"]
    
    # BWD 커널도 FWD와 동일한 Schema/Blob 구조를 기대함
    blob = struct.pack("<ii", int(ii), int(red))

    # 2. BWD 출력 Spec 정의 (Logits와 동일한 형상)
    logits_spec = b.values[logits_vid].spec
    dlogits_vid = b.value(f"{name}.dlogits", logits_spec)

    # 3. [핵심 수정] emit_resolved 호출
    # b 이후의 모든 인자는 'key=value' 형태여야 하며, ctx는 인자에 포함되지 않습니다.
    emit_resolved(
        b,
        kind="cross_entropy_bwd",
        name=name,
        inputs=[logits_vid, targets_vid, grad_y],
        outputs=[dlogits_vid],
        kind_id=ctx.CrossEntropyBwd,
        attr_schema=ctx.SCHEMA_XENT,
        attr_blob=blob,
        attrs={"ignore_index": ii, "reduction": red},
    )

    # 4. grad_map 갱신을 위한 반환
    return {logits_vid: dlogits_vid}