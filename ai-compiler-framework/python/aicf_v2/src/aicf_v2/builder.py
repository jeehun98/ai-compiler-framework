from __future__ import annotations
from typing import Any, Dict, List, Optional

from .tensor_spec import TensorSpec
from .graph import Value, Op, ValueRole


class Builder:
    def __init__(self, dtype: str, device: str):
        self.dtype = str(dtype)
        self.device = str(device)

        self.values: List[Value] = []
        self.ops: List[Op] = []

        self._name2vid: Dict[str, int] = {}

        # externals (feed로 들어오는 것들)
        self.input_vids: List[int] = []   # 순수 입력
        self.param_vids: List[int] = []   # W/b 같은 파라미터
        self.state_vids: List[int] = []   # optimizer state (m/v/step 등)

        # 기존 호환: "externals 전체"가 필요하면 이걸 쓰면 됨
        self.external_vids: List[int] = []

        # outputs (execution result interface)
        self.output_vids: List[int] = []
        self.outputs: Dict[str, int] = {}  # output alias name -> vid

    # -------- Values --------
    def input(self, name: str, spec: TensorSpec) -> int:
        vid = self._new_value(name, spec, producer_op=None, role="input")
        self.input_vids.append(vid)
        self.external_vids.append(vid)
        return vid

    def param(self, name: str, spec: TensorSpec) -> int:
        if name in self._name2vid:
            return self._name2vid[name]
        vid = self._new_value(name, spec, producer_op=None, role="param")
        self.param_vids.append(vid)
        self.external_vids.append(vid)
        return vid

    def state(self, name: str, spec: TensorSpec) -> int:
        if name in self._name2vid:
            return self._name2vid[name]
        vid = self._new_value(name, spec, producer_op=None, role="state")
        self.state_vids.append(vid)
        self.external_vids.append(vid)
        return vid

    def value(self, name: str, spec: TensorSpec) -> int:
        return self._new_value(name, spec, producer_op=None, role="tmp")

    def _new_value(self, name: str, spec: TensorSpec, producer_op: Optional[int], role: ValueRole) -> int:
        if name in self._name2vid:
            raise ValueError(f"Value name already exists: {name}")
        vid = len(self.values)
        self._name2vid[name] = vid
        self.values.append(Value(vid=vid, name=str(name), spec=spec, producer_op=producer_op, role=role))
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
        # ---- NEW: optional caches ----
        kind_id: Optional[int] = None,
        attr_schema: Optional[int] = None,
        attr_blob: Optional[bytes] = None,
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
            kind_id=None if kind_id is None else int(kind_id),
            attr_schema=None if attr_schema is None else int(attr_schema),
            attr_blob=attr_blob,
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

        if name in self.outputs and self.outputs[name] != vid:
            raise ValueError(
                f"Output name '{name}' already mapped to vid={self.outputs[name]}, cannot remap to vid={vid}"
            )

        self.outputs[name] = vid

        if vid not in self.output_vids:
            self.output_vids.append(vid)

        # 선택: output role로 바꾸고 싶으면
        # self.values[vid].role = "output"
