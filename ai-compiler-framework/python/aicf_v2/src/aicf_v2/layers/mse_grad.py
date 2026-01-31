from __future__ import annotations
from .base import Layer
from ..tensor_spec import TensorSpec


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

    def emit(self, b, pred: int, target: int) -> int:
        p = b.values[pred].spec
        t = b.values[target].spec

        if tuple(p.shape) != tuple(t.shape):
            raise ValueError(f"MseGrad shape mismatch: pred={p.shape} target={t.shape}")
        if p.dtype != t.dtype:
            raise ValueError(f"MseGrad dtype mismatch: pred={p.dtype} target={t.dtype}")
        if p.device != t.device:
            raise ValueError(f"MseGrad device mismatch: pred={p.device} target={t.device}")

        g = b.value(f"{self.name}.g", TensorSpec(shape=p.shape, dtype=p.dtype, device=p.device))

        if self.scale is None:
            kind = "mse_grad"
            attrs = {}
        else:
            kind = "mse_grad_scaled"
            attrs = {"scale": self.scale}

        b.emit(
            kind,
            inputs=[pred, target],
            outputs=[g],
            name=f"{self.name}.{kind}",
            attrs=attrs,
        )
        return g
