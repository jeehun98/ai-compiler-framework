# src/aicf_v2/layers/gemm_epilogue.py
from __future__ import annotations
from .base import Layer
from ..tensor_spec import TensorSpec

class GemmEpilogue(Layer):
    """
    A @ B (+bias) then optional relu, fused in kernel.

    Contract (matches binding test):
      inputs  = [A, B, bias]   bias shape: (N,)
      outputs = [C]            C shape: (M, N)
      attrs payload: <iii> (transA, transB, relu) with schema_id=0
    """
    def __init__(self, name: str, *, transA: bool = False, transB: bool = False, relu: bool = True):
        super().__init__(name)
        self.transA = bool(transA)
        self.transB = bool(transB)
        self.relu = bool(relu)

    def emit(self, b, A: int, B: int, bias: int) -> int:
        a = b.values[A].spec
        w = b.values[B].spec
        bi = b.values[bias].spec

        if len(a.shape) != 2 or len(w.shape) != 2:
            raise ValueError("GemmEpilogue: only 2D supported for now")

        # logical A2: (M,K), B2: (K,N)
        M, K = (a.shape[1], a.shape[0]) if self.transA else (a.shape[0], a.shape[1])
        K2, N = (w.shape[1], w.shape[0]) if self.transB else (w.shape[0], w.shape[1])

        if K2 != K:
            raise ValueError(f"GemmEpilogue: K mismatch A(K={K}) vs B(K={K2})")
        if bi.shape != (N,):
            raise ValueError(f"GemmEpilogue: bias must be (N,) where N={N}; got {bi.shape}")

        y_spec = TensorSpec(shape=(M, N), dtype=a.dtype, device=a.device)
        Y = b.value(f"{self.name}.out", y_spec)

        b.emit(
            "gemm_epilogue",
            inputs=[A, B, bias],
            outputs=[Y],
            name=f"{self.name}.gemm_epilogue",
            attrs={"transA": self.transA, "transB": self.transB, "relu": self.relu},
            hints={"prefer_epilogue_fusion": True},
        )
        return Y
