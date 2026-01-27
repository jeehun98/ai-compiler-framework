# examples/python/aicf_fw/core_v2/op_attrs/copy.py
from __future__ import annotations
from typing import Any, Optional
from .base import OpAttr, LoweredOpView, ValueDescs, fill_common

class CopyAttrBuilder:
    kind = "copy"

    def build(self, op: Any, value_descs: ValueDescs, op_id: Optional[int] = None) -> OpAttr:
        v = LoweredOpView.from_any(op, op_id=op_id)
        attr = OpAttr(
            op_kind=v.kind, op_id=v.op_id,
            inputs=v.inputs, outputs=v.outputs,
            params=dict(v.attrs), kid=v.kid
        )
        attr.sig = "COPY"
        fill_common(attr, v, value_descs)
        return attr


class CopySavedAttrBuilder(CopyAttrBuilder):
    kind = "copy_saved"
