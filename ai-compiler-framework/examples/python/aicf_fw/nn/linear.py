from __future__ import annotations
import torch
from aicf_fw.fw.module import Module
from aicf_fw.fw.naming import param_name
from aicf_fw.core_v2.ops import linear as ir_linear

class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        W = torch.randn(out_features, in_features, device=device, dtype=dtype) * 0.02
        self.register_parameter("W", W)
        if bias:
            b = torch.zeros(out_features, device=device, dtype=dtype)
            self.register_parameter("b", b)
        self.bias = bias

    def forward_ir(self, x_sym, psym: dict[str, object]):
        pfx = self._prefix
        W = psym[param_name(pfx, "W")]
        b = psym[param_name(pfx, "b")] if self.bias else None
        return ir_linear(x_sym, W, b, name=f"{pfx}.out" if pfx else "out")
