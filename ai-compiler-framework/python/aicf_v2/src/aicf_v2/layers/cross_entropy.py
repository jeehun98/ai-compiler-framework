from __future__ import annotations
from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda.cross_entropy import cross_entropy_fwd, cross_entropy_bwd

class CrossEntropyLoss(Layer):
    def __init__(self, ignore_index: int = -100, reduction: str = "mean", name: str = "xent"):
        super().__init__(name)
        self.ignore_index = ignore_index
        self.reduction_id = 0 if reduction == "mean" else 1

    def emit(self, b, logits: int, targets: int, *, ctx) -> int:
        out_spec = TensorSpec(shape=(1,), dtype="f32", device=b.device)
        out = b.value(f"{self.name}.out", out_spec)

        cross_entropy_fwd(
            b, ctx, logits=logits, targets=targets, out=out,
            ignore_index=self.ignore_index, reduction=self.reduction_id,
            name=f"{self.name}.fwd"
        )
        return out

    def emit_backward(self, b, ctx, inputs, outputs, grad_y, **kwargs) -> dict[str, int]:
        logits = inputs[0]
        targets = inputs[1]
        
        d_logits = b.value(f"{self.name}.d_logits", b.values[logits].spec)

        cross_entropy_bwd(
            b, ctx, 
            logits=logits, targets=targets, grad_out=grad_y, 
            out_dlogits=d_logits,
            ignore_index=self.ignore_index, reduction=self.reduction_id,
            name=f"{self.name}.bwd"
        )

        return {"input": d_logits}