# aicf_v2/models/sequential.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from aicf_v2.fw.module import Module
from aicf_v2.fw.emit_ctx import EmitCtx


@dataclass
class Sequential(Module):
    layers: List[Module] = field(default_factory=list)

    def __post_init__(self) -> None:
        Module.__init__(self)
        for i, m in enumerate(self.layers):
            self.add_module(str(i), m)

    def emit(self, ctx: EmitCtx, x_vid: int) -> int:
        cur = x_vid
        for i, m in enumerate(self.layers):
            cur = m.emit(ctx, cur)
        return cur
