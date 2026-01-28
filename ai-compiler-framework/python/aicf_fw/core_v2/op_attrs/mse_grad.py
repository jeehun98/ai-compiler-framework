# examples/python/aicf_fw/core_v2/op_attrs/mse_grad.py
from __future__ import annotations
from typing import Any, Optional
from .base import OpAttr, LoweredOpView, ValueDescs, fill_common

class MseGradAttrBuilder:
    kind = "mse_grad"

    def build(self, op: Any, value_descs: ValueDescs, op_id: Optional[int] = None) -> OpAttr:
        v = LoweredOpView.from_any(op, op_id=op_id)
        attr = OpAttr(
            op_kind=v.kind, op_id=v.op_id,
            inputs=v.inputs, outputs=v.outputs,
            params=dict(v.attrs), kid=v.kid
        )
        attr.sig = "MSE_GRAD"
        fill_common(attr, v, value_descs)

        # best-effort: record that output matches input0 shape typically
        attr.params.setdefault("out_matches_in0", True)
        return attr
