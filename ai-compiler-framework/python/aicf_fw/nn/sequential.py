# aicf_fw/nn/sequential.py
from __future__ import annotations

from aicf_fw.fw.module import Module
from aicf_fw.fw.emit_ctx import EmitCtx


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers: list[Module] = []
        for i, m in enumerate(layers):
            self.layers.append(m)
            self.add_module(str(i), m)

    def emit(self, ctx: EmitCtx, x_vid: int) -> int:
        out = x_vid
        for m in self.layers:
            if not hasattr(m, "emit"):
                raise RuntimeError(f"layer {m.__class__.__name__} has no emit(ctx, x_vid)")
            out = m.emit(ctx, out)
        return out

    # keep old API shape if something still calls forward_ir
    def forward_ir(self, x_sym, psym: dict[str, object]):
        out = x_sym
        for m in self.layers:
            out = m.forward_ir(out, psym)
        return out
