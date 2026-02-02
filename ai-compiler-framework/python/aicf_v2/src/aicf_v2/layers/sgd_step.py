from __future__ import annotations

from .base import Layer
from ..tensor_spec import TensorSpec

from ..emitters.cuda.context import CudaEmitContext
from ..emitters.cuda.sgd_step import sgd_step as emit_sgd_step


class SgdStep(Layer):
    """
    SGD step kernel wrapper.

    Contract:
      inputs : [P, G]
      outputs: [Pout]
      schema : 'SGDS'
      blob   : <f lr>
    """

    def __init__(self, name: str, *, lr: float = 1e-3):
        super().__init__(name)
        self.lr = float(lr)

    def emit(self, b, P: int, G: int, *, ctx: CudaEmitContext) -> int:
        P_spec = b.values[P].spec
        G_spec = b.values[G].spec

        if tuple(G_spec.shape) != tuple(P_spec.shape):
            raise ValueError(f"SgdStep shape mismatch: P={P_spec.shape} G={G_spec.shape}")
        if G_spec.dtype != P_spec.dtype:
            raise ValueError(f"SgdStep dtype mismatch: P={P_spec.dtype} G={G_spec.dtype}")
        if G_spec.device != P_spec.device:
            raise ValueError(f"SgdStep device mismatch: P={P_spec.device} G={G_spec.device}")

        # keep separate; planner may alias/inplace later
        Pout = b.value(f"{self.name}.P", TensorSpec(shape=P_spec.shape, dtype=P_spec.dtype, device=P_spec.device))

        emit_sgd_step(
            b, ctx,
            P=P,
            G=G,
            outP=Pout,
            lr=self.lr,
            name=f"{self.name}.sgd_step",
            constraints={"inplace_ok": True},
        )
        return Pout
