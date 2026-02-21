from __future__ import annotations
import struct
from typing import Dict, Any

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved
from . import mse_grad  # 미분 계산을 위해 mse_grad 모듈 참조

def emit(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    pred: int,
    target: int,
    out: int,
    reduction: str = "mean",
    name: str = "mse_loss"
) -> int:
    """MSE Loss Forward 연산을 IR에 기록합니다."""
    # 'MSEL' Schema ID 처리
    if reduction == "mean":
        schema_id = 0
        blob = b""
    elif reduction == "sum":
        schema_id = 0x4C45534D  # 'MSEL'
        blob = struct.pack("<i", 1)
    else:
        raise ValueError("Only 'mean' or 'sum' supported")

    return emit_resolved(
        b,
        kind="mse_loss",
        name=name,
        inputs=[pred, target],
        outputs=[out],
        kind_id=ctx.MseLoss,
        attr_schema=schema_id,
        attr_blob=blob,
        attrs={"reduction": reduction}
    )

def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,        # 최적화된 FWD MseLoss EmitNode
    grad_y: int,          # dy (보통 Loss에 대한 grad_initial)
    name: str = "mse_loss_bwd",
) -> Dict[int, int]:
    """
    Mirroring: FWD 노드의 속성을 읽어와서 대응하는 MSE Gradient 연산을 누적합니다.
    """
    pred_vid = fwd_node.inputs[0]
    target_vid = fwd_node.inputs[1]
    reduction = fwd_node.attrs.get("reduction", "mean")

    # 1. Reduction 방식에 따른 Scale 결정
    # mean인 경우 2/N, sum인 경우 2.0 (mse_grad 커널 규격에 따름)
    # 여기서는 mse_grad 모듈의 기본 로직을 사용하거나 scale을 명시합니다.
    p_spec = b.values[pred_vid].spec
    numel = 1
    for s in p_spec.shape:
        numel *= s
    
    scale = (2.0 / numel) if reduction == "mean" else 2.0

    # 2. Gradient Vid 생성
    g_pred = b.value(f"{name}.d_pred", p_spec)

    # 3. mse_grad 모듈을 호출하여 실제 미분 노드 누적
    mse_grad.emit(
        b, ctx,
        pred=pred_vid,
        target=target_vid,
        out=g_pred,
        scale=scale,
        name=f"{name}.mse_grad"
    )

    # 4. Pred에 대한 미분값 반환 (Target은 보통 미분하지 않음)
    return {pred_vid: g_pred}