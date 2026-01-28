# examples/python/aicf_fw/core_v2/op_attrs/sgd_step.py
from __future__ import annotations
from typing import Any, Optional
from .base import OpAttr, LoweredOpView, ValueDescs, fill_common

class SgdStepAttrBuilder:
    kind = "sgd_step"

    def build(self, op: Any, value_descs: ValueDescs, op_id: Optional[int] = None) -> OpAttr:
        v = LoweredOpView.from_any(op, op_id=op_id)
        attr = OpAttr(
            op_kind=v.kind, op_id=v.op_id,
            inputs=v.inputs, outputs=v.outputs,
            params=dict(v.attrs), kid=v.kid
        )
        attr.sig = "OPTIM_STEP"
        fill_common(attr, v, value_descs)

        attr.params.setdefault("optim", "sgd")
        if "lr" in v.attrs:
            attr.params["lr"] = float(v.attrs["lr"])

        # in-place update 여부 (로그: (W, dW)->(W))
        if len(v.inputs) >= 1 and len(v.outputs) >= 1:
            attr.params.setdefault("inplace", (v.inputs[0] == v.outputs[0]))

        return attr
