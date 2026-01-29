# aicf_fw/core_v2/backend_ops.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aicf_fw.core_v2.op_attrs.base import OpAttr


@dataclass
class BackendOp:
    """
    nn 템플릿/컴파일러 파이프라인 공용 최소 단위.
    - op_kind: 'gemm', 'bias_add', 'relu', ...
    - inputs/outputs: IR value id(int) 또는 fw value id
    - attrs: lowered dict의 attrs에 해당 (op별 파라미터)
    - kernel_id: StageB에서 채움
    - op_attr: OpAttrs 레이어 (의미/조건 표준화)
    """
    op_kind: str
    inputs: List[int]
    outputs: List[int]
    attrs: Dict[str, Any] = field(default_factory=dict)

    kernel_id: Optional[str] = None
    op_attr: Optional[OpAttr] = None

    # 디버그/추적용
    name: str = ""
    op_id: int = -1

    def to_lowered_dict(self) -> Dict[str, Any]:
        d = {
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
            inputs=[int(x) for x in d.get("inputs", [])],
            outputs=[int(y) for y in d.get("outputs", [])],
            attrs=dict(d.get("attrs", {}) or {}),
            kernel_id=d.get("kernel_id", None),
            name=str(d.get("name", "")),
        )
