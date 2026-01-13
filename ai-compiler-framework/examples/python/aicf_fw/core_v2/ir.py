from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class IRValue:
    id: int
    name: str
    shape: Tuple[int, ...]
    dtype: str
    device: str


@dataclass
class IRNode:
    op: str
    inputs: List[int]
    outputs: List[int]
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRGraph:
    name: str = "graph"
    values: Dict[int, IRValue] = field(default_factory=dict)
    nodes: List[IRNode] = field(default_factory=list)
    _next_vid: int = 0

    def new_value(self, *, name: str, shape: Tuple[int, ...], dtype: str, device: str) -> IRValue:
        vid = self._next_vid
        self._next_vid += 1
        v = IRValue(id=vid, name=str(name), shape=tuple(shape), dtype=str(dtype), device=str(device))
        self.values[vid] = v
        return v

    def emit(self, *, op: str, inputs: List[IRValue], outputs: List[IRValue], attrs: Dict[str, Any] | None = None) -> IRNode:
        n = IRNode(
            op=str(op),
            inputs=[int(v.id) for v in inputs],
            outputs=[int(v.id) for v in outputs],
            attrs=dict(attrs or {}),
        )
        self.nodes.append(n)
        return n
