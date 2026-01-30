from __future__ import annotations
from .base import Layer
from ..tensor_spec import TensorSpec

class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, name: str, bias: bool = True):
        super().__init__(name)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bias = bool(bias)

    def emit(self, b, x: int) -> int:
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
        b.emit(
            "gemm",
            inputs=[x, W],
            outputs=[y],
            name=f"{self.name}.gemm",
            attrs={"transA": False, "transB": True},
            hints={"prefer_epilogue_fusion": True},
        )

        if not self.bias:
            return y

        bias = b.param(
            f"{self.name}.b",
            TensorSpec(shape=(self.out_features,), dtype=b.dtype, device=b.device),
        )

        # 여기서 "inplace로 y에 더해라"를 박지 않고,
        # 플래너가 alias/inplace를 결정할 수 있도록 out을 분리해 둠.
        y2 = b.value(f"{self.name}.out_bias", y_spec)

        b.emit(
            "bias_add",
            inputs=[y, bias],
            outputs=[y2],
            name=f"{self.name}.bias_add",
            attrs={"broadcast_axis": -1},
            constraints={"inplace_ok": True},
            hints={"prefer_epilogue_fusion": True},
        )

        return y2
