# examples/python/aicf_fw/core_v2/op_attrs/gemm.py
from __future__ import annotations
from typing import Any, Optional
from .base import OpAttr, LoweredOpView, ValueDescs, fill_common

class GemmAttrBuilder:
    kind = "gemm"

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
        attr.sig = "GEMM"
        fill_common(attr, v, value_descs)

        # lowered attrs
        attr.layout["transA"] = bool(v.attrs.get("transA", v.attrs.get("trans_a", False)))
        attr.layout["transB"] = bool(v.attrs.get("transB", v.attrs.get("trans_b", False)))

        # optional: infer MNK (best-effort)
        x = attr.shapes.get("in0", ())
        w = attr.shapes.get("in1", ())
        if len(x) == 2 and len(w) == 2:
            # 네 convention: y = x @ W^T + b (transB=True)
            M = x[0]
            K = x[1]
            if attr.layout["transB"]:
                # W: (N, K)
                N = w[0]
            else:
                # W: (K, N)
                N = w[1]
            attr.params.setdefault("mnk", (int(M), int(N), int(K)))

        return attr
