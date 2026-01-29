# aicf_fw/fw/emit_ctx.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# core_v2 BackendOp만 쓰고 (이건 이미 있음)
from aicf_fw.core_v2.backend_ops import BackendOp


@dataclass
class ValueDesc:
    name: str
    shape: tuple[int, ...]
    dtype: Any
    device: Any
    role: str = "tmp"  # input/param/tmp/static/meta


@dataclass
class FrozenGraph:
    name: str
    values: List[ValueDesc]
    ops: List[BackendOp]


@dataclass
class EmitCtx:
    B: int
    D: int
    device: Any
    dtype: Any
    name: str = "emit_graph"

    values: List[ValueDesc] = field(default_factory=list)
    ops: List[BackendOp] = field(default_factory=list)

    producer: Dict[int, int] = field(default_factory=dict)   # value_id -> op_index
    name2vid: Dict[str, int] = field(default_factory=dict)

    # ----------------------------
    # value helpers
    # ----------------------------
    def new_value(self, name: str, shape: tuple[int, ...], role: str = "tmp") -> int:
        if name in self.name2vid:
            raise RuntimeError(f"[EmitCtx] value name already exists: {name}")
        vid = len(self.values)
        self.values.append(ValueDesc(name=name, shape=shape, dtype=self.dtype, device=self.device, role=role))
        self.name2vid[name] = vid
        return vid

    def get_vid(self, name: str) -> int:
        if name not in self.name2vid:
            raise KeyError(f"[EmitCtx] unknown value name: {name}")
        return int(self.name2vid[name])

    def input_vid(self, name: str, shape: tuple[int, ...]) -> int:
        if name in self.name2vid:
            return int(self.name2vid[name])
        return self.new_value(name=name, shape=shape, role="input")

    def param_vid(self, pname: str, shape: tuple[int, ...]) -> int:
        if pname in self.name2vid:
            return int(self.name2vid[pname])
        return self.new_value(name=pname, shape=shape, role="param")

    def static_vid(self, name: str, shape: tuple[int, ...]) -> int:
        if name in self.name2vid:
            return int(self.name2vid[name])
        return self.new_value(name=name, shape=shape, role="static")

    def meta_vid(self, name: str, shape: tuple[int, ...], role: str = "meta") -> int:
        if name in self.name2vid:
            return int(self.name2vid[name])
        return self.new_value(name=name, shape=shape, role=role)

    # ----------------------------
    # op emit
    # ----------------------------
    def emit_op(self, op_kind: str, inputs: list[int], outputs: list[int], name: str = "", **attrs) -> int:
        op = BackendOp(
            op_kind=str(op_kind),
            inputs=[int(x) for x in inputs],
            outputs=[int(y) for y in outputs],
            attrs=dict(attrs),
            name=str(name),
        )
        op_index = len(self.ops)
        self.ops.append(op)
        for o in outputs:
            self.producer[int(o)] = op_index
        return op_index

    def freeze(self, *, meta_as_static: bool = True) -> FrozenGraph:
        if meta_as_static:
            for vd in self.values:
                if vd.role == "meta":
                    vd.role = "static"
        return FrozenGraph(name=self.name, values=self.values, ops=self.ops)
