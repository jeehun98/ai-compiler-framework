from __future__ import annotations
from .base import Layer
from ..tensor_spec import TensorSpec


class ReluBwd(Layer):
    """
    ReluBwd:
      dY = (Y > 0) ? dOut : 0

    Kernel contract:
      inputs : [dOut, Y]   (order matters!)
      outputs: [dY]
      schema : 0
      payload: empty
    """

    def __init__(self, name: str):
        super().__init__(name)

    def emit(self, b, d_out: int, y: int) -> int:
        dy_spec = b.values[y].spec
        do_spec = b.values[d_out].spec

        if tuple(do_spec.shape) != tuple(dy_spec.shape):
            raise ValueError(f"ReluBwd shape mismatch: dOut={do_spec.shape} Y={dy_spec.shape}")
        if do_spec.dtype != dy_spec.dtype:
            raise ValueError(f"ReluBwd dtype mismatch: dOut={do_spec.dtype} Y={dy_spec.dtype}")
        if do_spec.device != dy_spec.device:
            raise ValueError(f"ReluBwd device mismatch: dOut={do_spec.device} Y={dy_spec.device}")

        dY = b.value(f"{self.name}.dY", TensorSpec(shape=dy_spec.shape, dtype=dy_spec.dtype, device=dy_spec.device))

        b.emit(
            "relu_bwd",
            inputs=[d_out, y],   # ✅ 반드시 이 순서
            outputs=[dY],
            name=f"{self.name}.relu_bwd",
        )
        return dY
