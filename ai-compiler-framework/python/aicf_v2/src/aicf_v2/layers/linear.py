from __future__ import annotations

from .base import Layer
from ..tensor_spec import TensorSpec

from ..emitters.cuda.context import CudaEmitContext
from ..emitters.cuda.gemm import gemm as emit_gemm
from ..emitters.cuda.bias_add import bias_add as emit_bias_add


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, name: str, bias: bool = True):
        super().__init__(name)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bias = bool(bias)

    def emit(self, b, x: int, *, ctx: CudaEmitContext) -> int:
        x_spec = b.values[x].spec
        if x_spec.shape[-1] != self.in_features:
            raise ValueError(f"Linear({self.in_features}->{self.out_features}) got x.shape[-1]={x_spec.shape[-1]}")

        # (*, in_features) -> (*, out_features)
        y_shape = (*x_spec.shape[:-1], self.out_features)
        y_spec = TensorSpec(shape=y_shape, dtype=x_spec.dtype, device=x_spec.device)
        y = b.value(f"{self.name}.out", y_spec)

        W = b.param(
            f"{self.name}.W",
            TensorSpec(shape=(self.out_features, self.in_features), dtype=b.dtype, device=b.device),
        )

        # y = x @ W^T
        emit_gemm(
            b, ctx,
            A=x,
            B=W,
            out=y,
            transA=False,
            transB=True,
            name=f"{self.name}.gemm",
            hints={"prefer_epilogue_fusion": True},
        )

        if not self.bias:
            return y

        bias = b.param(
            f"{self.name}.b",
            TensorSpec(shape=(self.out_features,), dtype=b.dtype, device=b.device),
        )

        # out 분리: planner가 alias/inplace 결정
        y2 = b.value(f"{self.name}.out_bias", y_spec)

        emit_bias_add(
            b, ctx,
            x=y,
            bias=bias,
            out=y2,
            broadcast_axis=-1,
            name=f"{self.name}.bias_add",
            constraints={"inplace_ok": True},
            hints={"prefer_epilogue_fusion": True},
        )

        return y2
