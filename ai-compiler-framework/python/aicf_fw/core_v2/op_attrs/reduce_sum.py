# examples/python/aicf_fw/core_v2/op_attrs/reduce_sum.py
from __future__ import annotations
from typing import Any, Optional
from .base import OpAttr, LoweredOpView, ValueDescs, fill_common

class ReduceSumAttrBuilder:
    kind = "reduce_sum"

    def build(self, op: Any, value_descs: ValueDescs, op_id: Optional[int] = None) -> OpAttr:
        v = LoweredOpView.from_any(op, op_id=op_id)
        attr = OpAttr(
            op_kind=v.kind, op_id=v.op_id,
            inputs=v.inputs, outputs=v.outputs,
            params=dict(v.attrs), kid=v.kid
        )
        attr.sig = "REDUCE_SUM"
        fill_common(attr, v, value_descs)

        # axis/keepdim (from your log)
        if "axis" in v.attrs:
            attr.params["axis"] = int(v.attrs["axis"])
        if "keepdim" in v.attrs:
            attr.params["keepdim"] = bool(v.attrs["keepdim"])

        # best-effort: infer expected output shape from input shape + axis/keepdim
        in0 = attr.shapes.get("in0", ())
        axis = attr.params.get("axis", None)
        keepdim = attr.params.get("keepdim", False)
        if isinstance(axis, int) and len(in0) > 0:
            ax = axis if axis >= 0 else axis + len(in0)
            if 0 <= ax < len(in0):
                if keepdim:
                    out_shape = tuple((1 if i == ax else in0[i]) for i in range(len(in0)))
                else:
                    out_shape = tuple(in0[i] for i in range(len(in0)) if i != ax)
                attr.params.setdefault("expected_out_shape", out_shape)

        return attr
