from __future__ import annotations

from typing import Callable

from .ir import IRGraph
from .trace import tracing


def trace_ir(step_fn: Callable[[], None], *, name: str = "train_step") -> IRGraph:
    ir = IRGraph(name=name)
    with tracing(ir):
        step_fn()
    return ir
