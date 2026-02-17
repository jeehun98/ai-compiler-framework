from __future__ import annotations
import struct
from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def mse_loss(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    pred: int,
    target: int,
    out: int,
    reduction: str = "mean",
    name: str = "mse_loss"
) -> int:
    # 커널 테스트의 로직 반영
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
        kind_id=ctx.MseLoss, # ID=20
        attr_schema=schema_id,
        attr_blob=blob,
        attrs={"reduction": reduction}
    )