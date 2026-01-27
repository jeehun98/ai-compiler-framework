# examples/python/aicf_fw/core_v2/op_attrs/relu.py
from __future__ import annotations
from typing import Any, Optional
from .base import OpAttr, LoweredOpView, ValueDescs, fill_common

class ReluAttrBuilder:
    kind = "relu"

    def build(self, op: Any, value_descs: ValueDescs, op_id: Optional[int] = None) -> OpAttr:
        v = LoweredOpView.from_any(op, op_id=op_id)
        attr = OpAttr(
            op_kind=v.kind, op_id=v.op_id,
            inputs=v.inputs, outputs=v.outputs,
            params=dict(v.attrs), kid=v.kid
        )
        attr.sig = "ACT"
        fill_common(attr, v, value_descs)

        # normalize act kind
        attr.params.setdefault("act", "relu")

        # in-place hint: your lowered relu writes (3)->(4), not same id, so default False
        attr.params.setdefault("inplace", (len(v.inputs) == 1 and len(v.outputs) == 1 and v.inputs[0] == v.outputs[0]))

        return attr
