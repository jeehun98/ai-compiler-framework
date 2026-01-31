from __future__ import annotations
from typing import Any, Dict, List, Optional

from .tensor_spec import TensorSpec
from .graph import Value, Op


class Builder:
    def __init__(self, dtype: str, device: str):
        self.dtype = str(dtype)
        self.device = str(device)

        self.values: List[Value] = []
        self.ops: List[Op] = []

        self._name2vid: Dict[str, int] = {}
        self.input_vids: List[int] = []

        # outputs (execution result interface)
        self.output_vids: List[int] = []
        self.outputs: Dict[str, int] = {}  # output alias name -> vid

    # -------- Values --------
    def input(self, name: str, spec: TensorSpec) -> int:
        vid = self._new_value(name, spec, producer_op=None)
        self.input_vids.append(vid)
        return vid

    def param(self, name: str, spec: TensorSpec) -> int:
        # params are "externally provided" -> treat as inputs
        if name in self._name2vid:
            return self._name2vid[name]
        vid = self._new_value(name, spec, producer_op=None)
        self.input_vids.append(vid)
        return vid

    def value(self, name: str, spec: TensorSpec) -> int:
        return self._new_value(name, spec, producer_op=None)

    def _new_value(self, name: str, spec: TensorSpec, producer_op: Optional[int]) -> int:
        if name in self._name2vid:
            raise ValueError(f"Value name already exists: {name}")
        vid = len(self.values)
        self._name2vid[name] = vid
        self.values.append(Value(vid=vid, name=str(name), spec=spec, producer_op=producer_op))
        return vid

    # -------- Emit ops --------
    def emit(
        self,
        kind: str,
        *,
        inputs: List[int],
        outputs: List[int],
        name: str,
        attrs: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        hints: Optional[Dict[str, Any]] = None,
        saved: Optional[List[int]] = None,
    ) -> int:
        op_index = len(self.ops)
        op = Op(
            kind=str(kind),
            name=str(name),
            inputs=list(inputs),
            outputs=list(outputs),
            attrs=dict(attrs or {}),
            constraints=dict(constraints or {}),
            hints=dict(hints or {}),
            saved=list(saved or []),
        )
        self.ops.append(op)

        # book-keeping
        for out_vid in outputs:
            self.values[out_vid].producer_op = op_index
        for in_vid in inputs:
            self.values[in_vid].users.append(op_index)

        return op_index

    # -------- Outputs --------
    def output(self, name: str, vid: int) -> None:
        """
        Register a user-facing output alias.

        - `name` is the key the executor should return (out[name] = tensor).
        - `vid` is the internal value id.

        Keeps:
          - outputs[name] = vid
          - output_vids maintains unique vids in insertion order
        """
        name = str(name)
        vid = int(vid)

        # optional: prevent accidental remap of the same output name to a different vid
        if name in self.outputs and self.outputs[name] != vid:
            raise ValueError(f"Output name '{name}' already mapped to vid={self.outputs[name]}, cannot remap to vid={vid}")

        self.outputs[name] = vid

        # keep deterministic unique list of vids
        if vid not in self.output_vids:
            self.output_vids.append(vid)
