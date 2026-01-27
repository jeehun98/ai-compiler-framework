# examples/python/aicf_fw/core_v2/op_attrs/bias_add.py
from __future__ import annotations
from typing import Any, Optional
from .base import OpAttr, LoweredOpView, ValueDescs, fill_common

class BiasAddAttrBuilder:
    kind = "bias_add"

    def build(self, op: Any, value_descs: ValueDescs, op_id: Optional[int] = None) -> OpAttr:
        v = LoweredOpView.from_any(op, op_id=op_id)
        attr = OpAttr(
            op_kind=v.kind,
            op_id=v.op_id,
            inputs=v.inputs,
            outputs=v.outputs,
            params=dict(v.attrs),
            kid=v.kid,
        )
        attr.sig = "BIAS_ADD"
        fill_common(attr, v, value_descs)

        # best-effort: bias shape 기록
        bias_shape = attr.shapes.get("in1", ())
        attr.params.setdefault("bias_shape", bias_shape)
        return attr
