from __future__ import annotations
from typing import List, Dict

from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda.context import CudaEmitContext
from ..emitters.cuda.softmax import softmax as emit_softmax
from ..emitters.cuda.softmax_bwd import softmax_bwd as emit_softmax_bwd

class Softmax(Layer):
    def __init__(self, axis: int = -1, name: str = "softmax"):
        super().__init__(name)
        self.axis = axis

    def emit(self, b, x: int, *, ctx: CudaEmitContext) -> int:
        """Forward: y = softmax(x)"""
        x_spec = b.values[x].spec
        y_spec = TensorSpec(shape=x_spec.shape, dtype=x_spec.dtype, device=x_spec.device)
        y = b.value(f"{self.name}.out", y_spec)

        emit_softmax(
            b, ctx,
            x=x,
            out=y,
            axis=self.axis,
            name=self.name
        )
        return y

    def emit_backward(self, b, ctx: CudaEmitContext, inputs: List[int], outputs: List[int], 
                      grad_y: int, params: List[int] = None, **kwargs) -> Dict[str, int]:
        """
        Softmax Backward (Emitter 인자명 매핑 완료):
        - out: Forward의 출력 y (outputs[0])
        - grad_out: 상위 레이어의 미분값 dy (grad_y)
        - grad_in: 계산되어 나갈 미분값 dx
        """
        y_vid = outputs[0]      # Forward 시의 softmax 결과
        dy_vid = grad_y         # Loss로부터 온 미분값
        
        y_spec = b.values[y_vid].spec
        dx_vid = b.value(f"{self.name}.dx", y_spec)

        # 이미터 인자 규격에 맞춰 호출: out, grad_out, grad_in
        emit_softmax_bwd(
            b, ctx,
            out=y_vid,          # 이미터의 'out' 인자에 y 주입
            grad_out=dy_vid,    # 이미터의 'grad_out' 인자에 dy 주입
            grad_in=dx_vid,     # 이미터의 'grad_in' 인자에 결과 dx 주입
            axis=self.axis,
            name=f"{self.name}.bwd"
        )

        return {"input": dx_vid}