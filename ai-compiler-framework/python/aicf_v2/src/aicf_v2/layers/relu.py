from __future__ import annotations
from .base import Layer
from ..tensor_spec import TensorSpec

from ..emitters.cuda.context import CudaEmitContext
from ..emitters.cuda.relu import relu as emit_relu
from ..emitters.cuda.relu_bwd import relu_bwd as emit_relu_bwd

class ReLU(Layer):
    def __init__(self, name: str):
        super().__init__(name)

    def emit(self, b, x: int, *, ctx: CudaEmitContext) -> int:
        """Forward: y = relu(x)"""
        x_spec = b.values[x].spec
        y = b.value(f"{self.name}.out", x_spec)

        # ReLU 커널 호출 (KID 자동 매핑)
        emit_relu(
            b, ctx,
            x=x,
            out=y,
            name=f"{self.name}.relu",
        )
        return y

    def emit_backward(self, b, ctx: CudaEmitContext, inputs: list[int], outputs: list[int], grad_y: int, params: list[int] = None, **kwargs) -> dict[str, int]:
        """
        ReLU Bwd: dx = dy * (y > 0)
        - outputs[0]: Forward 출력 y
        - grad_y: 상위 미분값 dy
        """
        y_vid = outputs[0]
        dy_vid = grad_y
        
        dy_spec = b.values[dy_vid].spec
        dx_vid = b.value(f"{self.name}.dx", dy_spec)

        # 이미 준비된 relu_bwd 이미터 호출
        emit_relu_bwd(
            b, ctx,
            dy=dy_vid,
            y=y_vid,
            out_dx=dx_vid,
            name=f"{self.name}.relu_bwd",
        )

        return {"input": dx_vid}