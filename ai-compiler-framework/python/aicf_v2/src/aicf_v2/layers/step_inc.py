from __future__ import annotations
from typing import TYPE_CHECKING

from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda import step_inc  # 통합 모듈 임포트

if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext

class StepInc(Layer):
    """
    학습 스텝 증가 레이어.
    Contract: step = step + 1 (i32 scalar)
    """
    def __init__(self, name: str):
        super().__init__(name)

    def emit(self, b, step: int, *, ctx: CudaEmitContext) -> int:
        ss = b.values[step].spec
        
        # v2 규약: 0d 미지원으로 인한 (1,) 형상 검증
        if ss.dtype != "i32":
            raise ValueError(f"StepInc expects i32 step; got {ss.dtype}")
        if tuple(ss.shape) != (1,):
            raise ValueError(f"StepInc expects shape (1,); got {ss.shape}")

        # 출력 Vid 생성 (Lattice: 입력과 동일한 Spec)
        out_step = b.value(f"{self.name}.out", TensorSpec(shape=(1,), dtype="i32", device=ss.device))

        # 통합된 step_inc.emit 호출
        step_inc.emit(
            b, ctx,
            step=step,
            out_step=out_step,
            name=f"{self.name}.step_inc",
            constraints={"inplace_ok": True},
        )
        return out_step