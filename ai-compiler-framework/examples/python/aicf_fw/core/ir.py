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
    """
    NOTE:
      - IRValue.name는 실행/바인딩 ABI로도 쓰이므로 "유니크"해야 함.
      - 기존에는 W/b 같은 이름이 중복되어 IR-only 실행에서 bind 불가였음.
      - 여기서 name uniquify를 강제한다.
    """
    def __init__(self, name: str = "graph"):
        self.name = name
        self._next_val_id = 0
        self._next_node_id = 0
        self.values: Dict[int, IRValue] = {}
        self.nodes: List[IRNode] = []

        # name uniquifier: "W" -> "W", "W#1", "W#2" ...
        self._name_count: Dict[str, int] = {}

    def _uniq_name(self, name: str) -> str:
        base = str(name)
        c = self._name_count.get(base, 0)
        self._name_count[base] = c + 1
        return base if c == 0 else f"{base}#{c}"

    def new_value(self, *, name: str, shape: Tuple[int, ...], dtype: str, device: str) -> IRValue:
        vid = self._next_val_id
        self._next_val_id += 1

        nm = self._uniq_name(name)

        v = IRValue(
            id=vid,
            name=nm,
            shape=tuple(shape),
            dtype=str(dtype),
            device=str(device),
        )
        self.values[vid] = v
        return v

    def emit(
        self,
        *,
        op: str,
        inputs: List[IRValue],
        outputs: List[IRValue],
        attrs: Optional[Dict[str, Any]] = None,
    ) -> IRNode:
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
