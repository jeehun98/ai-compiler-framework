# aicf_fw/core_v2/backend_ops.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Value / Op (source of truth)
# ----------------------------
@dataclass
class ValueDesc:
    """
    IR-lite value descriptor.
    - role: input/param/tmp/static/meta
    """
    name: str
    shape: Tuple[int, ...]
    dtype: Any
    device: Any
    role: str = "tmp"


@dataclass
class BackendOp:
    """
    Minimal common op unit shared by nn templates + compiler passes.

    - op_kind: 'gemm', 'bias_add', 'relu', ...
    - inputs/outputs: value ids (int)
    - attrs: op-specific params (lowered dict attrs equivalent)
    - kernel_id: filled by StageB
    - name: debug label (optional)
    """
    op_kind: str
    inputs: List[int]
    outputs: List[int]
    attrs: Dict[str, Any] = field(default_factory=dict)

    kernel_id: Optional[str] = None
    name: str = ""

    def to_lowered_dict(self) -> Dict[str, Any]:
        """
        Adapter for legacy passes that still expect dict-based lowered ops.
        Keep keys compatible with your current codebase:
          - 'op' and 'kind' both present
          - 'kernel_id' optional
          - 'attrs' always present
        """
        d: Dict[str, Any] = {
            "op": self.op_kind,
            "kind": self.op_kind,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": dict(self.attrs),
        }
        if self.kernel_id is not None:
            d["kernel_id"] = self.kernel_id
        if self.name:
            d["name"] = self.name
        return d

    @staticmethod
    def from_lowered_dict(d: Dict[str, Any]) -> "BackendOp":
        op_kind = d.get("kind", d.get("op", ""))
        return BackendOp(
            op_kind=str(op_kind),
            inputs=[int(x) for x in (d.get("inputs", []) or [])],
            outputs=[int(y) for y in (d.get("outputs", []) or [])],
            attrs=dict(d.get("attrs", {}) or {}),
            kernel_id=d.get("kernel_id", None),
            name=str(d.get("name", "")),
        )


# ----------------------------
# Frozen program (IR-lite)
# ----------------------------
@dataclass
class FrozenGraph:
    """
    What fw/compile produces after model.emit():
      - values: value table
      - ops: backend op stream
      - producer: value_id -> op_index
    """
    name: str
    values: List[ValueDesc]
    ops: List[BackendOp]
    producer: Dict[int, int] = field(default_factory=dict)

    def to_lowered_dicts(self) -> List[Dict[str, Any]]:
        return [op.to_lowered_dict() for op in self.ops]

    @staticmethod
    def from_lowered_dicts(
        *,
        name: str,
        values: List[ValueDesc],
        lowered: List[Dict[str, Any]],
        producer: Optional[Dict[int, int]] = None,
    ) -> "FrozenGraph":
        ops = [BackendOp.from_lowered_dict(d) for d in lowered]
        return FrozenGraph(name=name, values=values, ops=ops, producer=(producer or {}))


# ----------------------------
# Graph meta (needed for fuse/inplace/plan)
# ----------------------------
@dataclass
class GraphMeta:
    """
    Compile-time meta derived from values + ops.
    - use_count[v]: number of times v appears in any op input
    - last_use[v]: last op index that reads v
    """
    use_count: List[int]
    last_use: List[int]

    def use(self, v: int) -> int:
        return self.use_count[v]

    def last(self, v: int) -> int:
        return self.last_use[v]


def analyze_graph_meta(values: List[ValueDesc], ops: List[BackendOp]) -> GraphMeta:
    n = len(values)
    use_count = [0] * n
    last_use = [-1] * n

    for i, op in enumerate(ops):
        for v in op.inputs:
            if v < 0 or v >= n:
                raise IndexError(f"value id out of range: v={v} n_values={n} op#{i} kind={op.op_kind}")
            use_count[v] += 1
            last_use[v] = i

    return GraphMeta(use_count=use_count, last_use=last_use)


# ----------------------------
# Utilities
# ----------------------------
def assign_op_names_and_ids(ops: List[BackendOp]) -> None:
    """
    Optional helper to stamp stable ids / names for debugging.
    (If name already set, keep it.)
    """
    for i, op in enumerate(ops):
        # don't overwrite custom names
        if not op.name:
            op.name = f"{i:03d}:{op.op_kind}"
        # store id in attrs for trace/debug if you want (keeps BackendOp minimal)
        op.attrs.setdefault("_op_id", i)
