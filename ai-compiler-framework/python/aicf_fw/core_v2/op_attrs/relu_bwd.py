# examples/python/aicf_fw/core_v2/op_attrs/relu_bwd.py
from __future__ import annotations
from typing import Any, Optional
from .base import OpAttr, LoweredOpView, ValueDescs, fill_common

class ReluBwdAttrBuilder:
    kind = "relu_bwd"

    def build(self, op: Any, value_descs: ValueDescs, op_id: Optional[int] = None) -> OpAttr:
        v = LoweredOpView.from_any(op, op_id=op_id)
        attr = OpAttr(
            op_kind=v.kind, op_id=v.op_id,
            inputs=v.inputs, outputs=v.outputs,
            params=dict(v.attrs), kid=v.kid
        )
        attr.sig = "ACT_BWD"
        fill_common(attr, v, value_descs)

        # relu_bwd usually needs saved activation/mask
        # in your log: (d_relu0_out, relu0_saved) -> (d_lin0_out)
        attr.params.setdefault("act", "relu")
        attr.params.setdefault("needs_saved", True)

        return attr
