# python/aicf_v2/src/aicf_v2/builder.py

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
        self.external_vids: List[int] = []

        # outputs
        self.output_vids: List[int] = []
        self.outputs: Dict[str, int] = {} 

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
        """내부 연산 결과물을 위한 임시 Value 생성"""
        return self._new_value(name, spec, producer_op=None, role="tmp")

    def tensor_like(self, vid: int, name: Optional[str] = None) -> TensorSpec:
        """[Fix] 기존 Vid의 속성(Shape, Dtype 등)을 복사한 Spec 반환"""
        spec = self.values[vid].spec
        return TensorSpec(shape=spec.shape, dtype=spec.dtype, device=spec.device)

    def _new_value(self, name: str, spec: TensorSpec, producer_op: Optional[int], role: ValueRole) -> int:
        if name in self._name2vid:
            # 중복 이름 방지 (익명 이름 생성 로직을 추가할 수도 있음)
            name = f"{name}_{len(self.values)}"
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

        for out_vid in outputs:
            self.values[out_vid].producer_op = op_index
        for in_vid in inputs:
            self.values[in_vid].users.append(op_index)

        return op_index

    # -------- 편의 메서드 (테스트용) --------
    def op(self, kind: str, inputs: List[int], outputs: Any, name: Optional[str] = None) -> int:
        """[New] 테스트 코드에서 간단히 Op를 추가하기 위한 헬퍼"""
        if name is None:
            name = f"{kind}_{len(self.ops)}"
        
        # outputs가 Spec 리스트인 경우 자동으로 Value 생성
        actual_outputs = []
        for i, out in enumerate(outputs):
            if isinstance(out, TensorSpec):
                out_vid = self.value(f"{name}_out_{i}", out)
                actual_outputs.append(out_vid)
            else:
                actual_outputs.append(out)
        
        self.emit(kind, inputs=inputs, outputs=actual_outputs, name=name)
        return actual_outputs[0] if len(actual_outputs) == 1 else actual_outputs

    # -------- Outputs --------
    def output(self, name: str, vid: int) -> None:
        name = str(name)
        vid = int(vid)
        if name in self.outputs and self.outputs[name] != vid:
            raise ValueError(f"Output name '{name}' already mapped to vid={self.outputs[name]}")
        self.outputs[name] = vid
        if vid not in self.output_vids:
            self.output_vids.append(vid)