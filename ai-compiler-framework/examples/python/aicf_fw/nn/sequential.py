from __future__ import annotations
from aicf_fw.fw.module import Module

class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers: list[Module] = []
        for i, m in enumerate(layers):
            self.layers.append(m)
            self.add_module(str(i), m)

    def forward_ir(self, x_sym, psym: dict[str, object]):
        out = x_sym
        for m in self.layers:
            out = m.forward_ir(out, psym)
        return out
