from __future__ import annotations
from .base import Layer
from ..tensor_spec import TensorSpec


class AdamStep(Layer):
    """
    Adam step kernel wrapper.

    Contract (matches your _C.op_call test):
      inputs : P, G, M, V, bc1, bc2
      outputs: Pout, Mout, Vout
      attr_blob: <ffff = lr, beta1, beta2, eps
      schema: 'ADAM'
    Notes:
      - bc1, bc2 are scalar tensors (rank0) on same device/dtype as P
      - outputs are separate values; planner may alias/inplace later.
    """

    def __init__(
        self,
        name: str,
        *,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(name)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

    def emit(self, b, P: int, G: int, M: int, V: int, bc1: int, bc2: int):
        P_spec = b.values[P].spec
        G_spec = b.values[G].spec
        M_spec = b.values[M].spec
        V_spec = b.values[V].spec
        bc1_spec = b.values[bc1].spec
        bc2_spec = b.values[bc2].spec

        # --- basic checks (keep strict for bring-up) ---
        if P_spec.dtype != "f32":
            raise ValueError(f"AdamStep expects f32 params; got P.dtype={P_spec.dtype}")
        for nm, s in [("G", G_spec), ("M", M_spec), ("V", V_spec)]:
            if s.dtype != P_spec.dtype:
                raise ValueError(f"AdamStep dtype mismatch: P={P_spec.dtype} {nm}={s.dtype}")
            if s.device != P_spec.device:
                raise ValueError(f"AdamStep device mismatch: P={P_spec.device} {nm}={s.device}")
            if tuple(s.shape) != tuple(P_spec.shape):
                raise ValueError(f"AdamStep shape mismatch: P.shape={P_spec.shape} {nm}.shape={s.shape}")

        # bc1/bc2 must be scalar (v2: allow (1,) since 0d is forbidden)
        for nm, s in [("bc1", bc1_spec), ("bc2", bc2_spec)]:
            if s.dtype != P_spec.dtype:
                raise ValueError(f"AdamStep dtype mismatch: P={P_spec.dtype} {nm}={s.dtype}")
            if s.device != P_spec.device:
                raise ValueError(f"AdamStep device mismatch: P={P_spec.device} {nm}={s.device}")
            if tuple(s.shape) not in (tuple(()), (1,)):
                raise ValueError(f"AdamStep expects {nm} as scalar tensor; got shape={s.shape}")

        # outputs (same spec as P/M/V)
        Pout = b.value(f"{self.name}.P", P_spec)
        Mout = b.value(f"{self.name}.M", M_spec)
        Vout = b.value(f"{self.name}.V", V_spec)

        b.emit(
            "adam_step",
            inputs=[P, G, M, V, bc1, bc2],
            outputs=[Pout, Mout, Vout],
            name=f"{self.name}.adam_step",
            attrs={
                "lr": self.lr,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "eps": self.eps,
            },
            # planner가 inplace/alias를 결정하게 하고 싶으면
            # 우선 "가능"하다는 힌트만 둠.
            constraints={"inplace_ok": True},
        )

        return Pout, Mout, Vout
