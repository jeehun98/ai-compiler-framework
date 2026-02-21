from __future__ import annotations
from typing import TYPE_CHECKING

from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda import cross_entropy  # 통합된 모듈 임포트

if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext

# aicf_v2/layers/cross_entropy.py

class CrossEntropyLoss(Layer):
    def __init__(self, reduction: str = "mean", ignore_index: int = -100, name: str = "xent"):
        super().__init__(name=name)
        # 1. 여기서 필드명을 확실히 저장합니다.
        self.reduction = reduction 
        self.ignore_index = ignore_index

    def emit(self, b, logits_vid, targets_vid, ctx, out_spec=None, **kwargs):
        from ..emitters.cuda import cross_entropy as emit_xent
        from ..tensor_spec import TensorSpec

        # 출력 Spec 설정 (앞서 협의한 대로 [1] 형상 강제)
        if out_spec is None:
            out_spec = TensorSpec(shape=(1,), dtype="f32", device=b.device)
        
        out_vid = b.value(f"{self.name}.out", out_spec)

        # 2. self.reduction 값을 읽어 Emitter에 전달
        # 만약 self.reduction_id 같은 이름을 쓰고 싶다면 아래도 통일해야 합니다.
        reduction_mode = 0 if self.reduction == "mean" else 1

        emit_xent.emit(
            b, ctx,
            logits=logits_vid,
            targets=targets_vid,
            out=out_vid,
            ignore_index=self.ignore_index,
            reduction=reduction_mode,
            name=self.name
        )

        return out_vid