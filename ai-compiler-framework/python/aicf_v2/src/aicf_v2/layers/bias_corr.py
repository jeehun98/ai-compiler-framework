from __future__ import annotations
from .base import Layer
from ..tensor_spec import TensorSpec


class BiasCorr(Layer):
    """
    BiasCorr:
      (step:int32) -> (bc1_inv:float32, bc2_inv:float32)

    Kernel contract:
      inputs : [step]   (int32 scalar; v2 uses shape=(1,))
      outputs: [bc1_inv, bc2_inv] (float32)
      schema : 'BCOR'
      payload: <ff> beta1, beta2
    """

    def __init__(self, name: str, *, beta1: float = 0.9, beta2: float = 0.999):
        super().__init__(name)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)

    def emit(self, b, step: int) -> tuple[int, int]:
        ss = b.values[step].spec
        if ss.dtype != "i32":
            raise ValueError(f"BiasCorr expects step dtype i32; got {ss.dtype}")
        # v2: TensorSpec forbids 0d, so enforce rank1 scalar
        if tuple(ss.shape) != (1,):
            raise ValueError(f"BiasCorr expects step shape (1,) in v2; got {ss.shape}")

        o_spec = TensorSpec(shape=(1,), dtype="f32", device=ss.device)
        bc1 = b.value(f"{self.name}.bc1_inv", o_spec)
        bc2 = b.value(f"{self.name}.bc2_inv", o_spec)

        b.emit(
            "bias_corr",
            inputs=[step],
            outputs=[bc1, bc2],
            name=f"{self.name}.bias_corr",
            attrs={"beta1": self.beta1, "beta2": self.beta2},
        )
        return bc1, bc2
