from __future__ import annotations

from .base import Layer
from ..tensor_spec import TensorSpec

from ..emitters.cuda.context import CudaEmitContext
from ..emitters.cuda.mse_grad import mse_grad as emit_mse_grad


class MseGrad(Layer):
    """
    MseGrad:
      g = (pred - target) * scale

    scale:
      - None: kernel default scale = 2/numel (schema=0)
      - float: explicit scale via schema 'MSEG' + payload <f scale>

    inputs : [pred, target]
    outputs: [g]  (same shape/dtype/device as pred)
    """

    def __init__(self, name: str, *, scale: float | None = None):
        super().__init__(name)
        self.scale = None if scale is None else float(scale)

    def emit(self, b, pred: int, target: int, *, ctx: CudaEmitContext) -> int:
        p = b.values[pred].spec
        t = b.values[target].spec

        if tuple(p.shape) != tuple(t.shape):
            raise ValueError(f"MseGrad shape mismatch: pred={p.shape} target={t.shape}")
        if p.dtype != t.dtype:
            raise ValueError(f"MseGrad dtype mismatch: pred={p.dtype} target={t.dtype}")
        if p.device != t.device:
            raise ValueError(f"MseGrad device mismatch: pred={p.device} target={t.device}")

        g = b.value(f"{self.name}.g", TensorSpec(shape=p.shape, dtype=p.dtype, device=p.device))

        # ✅ kind/schema/blob 결정은 emitter로 위임
        emit_mse_grad(
            b, ctx,
            pred=pred,
            target=target,
            out=g,
            scale=self.scale,
            name=f"{self.name}.mse_grad",
        )
        return g
