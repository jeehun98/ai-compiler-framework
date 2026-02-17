from __future__ import annotations
import struct
from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def cross_entropy_fwd(b: Builder, ctx: CudaEmitContext, logits: int, targets: int, out: int, 
                      ignore_index: int = -100, reduction: int = 0, name: str = "xent_fwd"):
    # 속성 패킹
    blob = struct.pack("<ii", int(ignore_index), int(reduction))
    
    return emit_resolved(
        b, 
        kind="cross_entropy_fwd", 
        name=name,
        inputs=[logits, targets], 
        outputs=[out],
        kind_id=ctx.CrossEntropyFwd, 
        attr_schema=ctx.SCHEMA_XENT, # ✅ 추가: SCHEMA_XENT (fourcc("XENT"))
        attr_blob=blob,
        attrs={"ignore_index": ignore_index, "reduction": reduction}
    )

def cross_entropy_bwd(b: Builder, ctx: CudaEmitContext, logits: int, targets: int, grad_out: int, out_dlogits: int,
                      ignore_index: int = -100, reduction: int = 0, name: str = "xent_bwd"):
    blob = struct.pack("<ii", int(ignore_index), int(reduction))
    
    return emit_resolved(
        b, 
        kind="cross_entropy_bwd", 
        name=name,
        inputs=[logits, targets, grad_out], 
        outputs=[out_dlogits],
        kind_id=ctx.CrossEntropyBwd, 
        attr_schema=ctx.SCHEMA_XENT, # ✅ 추가: SCHEMA_XENT
        attr_blob=blob,
        attrs={"ignore_index": ignore_index, "reduction": reduction}
    )