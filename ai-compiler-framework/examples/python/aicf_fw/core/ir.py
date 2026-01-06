# aicf_fw/core/ir.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import json


@dataclass
class IRValue:
    id: int
    name: str
    shape: Tuple[int, ...]
    dtype: str
    device: str


@dataclass
class IRNode:
    id: int
    op: str
    inputs: List[int]          # IRValue ids
    outputs: List[int]         # IRValue ids
    attrs: Dict[str, Any]


class IRGraph:
    def __init__(self, name: str = "graph"):
        self.name = name
        self._next_val_id = 0
        self._next_node_id = 0
        self.values: Dict[int, IRValue] = {}
        self.nodes: List[IRNode] = []

    def new_value(self, *, name: str, shape: Tuple[int, ...], dtype: str, device: str) -> IRValue:
        vid = self._next_val_id
        self._next_val_id += 1
        v = IRValue(id=vid, name=name, shape=tuple(shape), dtype=str(dtype), device=str(device))
        self.values[vid] = v
        return v

    def emit(self, *, op: str, inputs: List[IRValue], outputs: List[IRValue], attrs: Optional[Dict[str, Any]] = None) -> IRNode:
        nid = self._next_node_id
        self._next_node_id += 1
        node = IRNode(
            id=nid,
            op=str(op),
            inputs=[v.id for v in inputs],
            outputs=[v.id for v in outputs],
            attrs=dict(attrs or {}),
        )
        self.nodes.append(node)
        return node

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph": self.name,
            "values": {str(k): asdict(v) for k, v in self.values.items()},
            "nodes": [asdict(n) for n in self.nodes],
        }

    def dump_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)
