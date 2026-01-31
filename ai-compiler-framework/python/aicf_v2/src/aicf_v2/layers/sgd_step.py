from __future__ import annotations
from .base import Layer
from ..tensor_spec import TensorSpec


class SgdStep(Layer):
    """
    inputs : P, G
    outputs: O  (P - lr * G)
    schema : 'SGDS'
    blob   : <f  (lr float32)

    Note:
      - O may alias P (in-place allowed)
      - O must NOT alias G (kernel rejects)
    """

    def __init__(self, name: str, *, lr: float = 1e-3):
        super().__init__(name)
        self.lr = float(lr)

    def emit(self, b, P: int, G: int) -> int:
        P_spec = b.values[P].spec
        G_spec = b.values[G].spec

        # dtype/device/shape checks
        if P_spec.dtype not in ("f16", "f32"):
            raise ValueError(f"SgdStep expects f16/f32; got P.dtype={P_spec.dtype}")
        if G_spec.dtype != P_spec.dtype:
            raise ValueError(f"SgdStep dtype mismatch: P={P_spec.dtype} G={G_spec.dtype}")
        if G_spec.device != P_spec.device:
            raise ValueError(f"SgdStep device mismatch: P={P_spec.device} G={G_spec.device}")
        if tuple(G_spec.shape) != tuple(P_spec.shape):
            raise ValueError(f"SgdStep shape mismatch: P.shape={P_spec.shape} G.shape={G_spec.shape}")

        O_spec = TensorSpec(shape=P_spec.shape, dtype=P_spec.dtype, device=P_spec.device)
        O = b.value(f"{self.name}.out", O_spec)

        b.emit(
            "sgd_step",
            inputs=[P, G],
            outputs=[O],
            name=f"{self.name}.sgd_step",
            attrs={"lr": self.lr},
            constraints={"inplace_ok": True},
        )
        return O
