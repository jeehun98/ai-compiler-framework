from __future__ import annotations
from aicf_fw.fw.module import Module
from aicf_fw.core_v2.ops import relu as ir_relu

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward_ir(self, x_sym, psym: dict[str, object]):
        return ir_relu(x_sym, name=f"{self._prefix}.relu_out" if self._prefix else "relu_out")
