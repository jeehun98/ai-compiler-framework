# examples/python/aicf_fw/core_v2/op_attrs/gemm_epilogue.py
from __future__ import annotations

from typing import Any, Optional
from .base import OpAttr, LoweredOpView, ValueDescs, fill_common


def _infer_epilogue_flags(v: LoweredOpView, attr: OpAttr) -> None:
    """
    StageC(fuse)에서 attrs에 epilogue_* 를 넣어준다는 전제.
    - epilogue_bias: bool
    - epilogue_act: str | None  (ex: "relu")
    """
    ep_bias = v.attrs.get("epilogue_bias", None)
    ep_act = v.attrs.get("epilogue_act", None)

    if ep_bias is not None:
        attr.params.setdefault("epilogue_bias", bool(ep_bias))
    else:
        # bias input이 3개째로 붙으면 bias=True로 추정
        # (x, w, b) 형태
        if len(v.inputs) >= 3:
            attr.params.setdefault("epilogue_bias", True)
        else:
            attr.params.setdefault("epilogue_bias", False)

    if ep_act is not None:
        # normalize
        attr.params.setdefault("epilogue_act", str(ep_act).lower())
    else:
        attr.params.setdefault("epilogue_act", None)


class GemmEpilogueAttrBuilder:
    kind = "gemm_epilogue"

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
        attr.sig = "GEMM_EPILOGUE"
        fill_common(attr, v, value_descs)

        # keep gemm layout flags (transA/transB)
        transA = bool(v.attrs.get("transA", v.attrs.get("trans_a", False)))
        transB = bool(v.attrs.get("transB", v.attrs.get("trans_b", False)))
        attr.layout["transA"] = transA
        attr.layout["transB"] = transB

        # epilogue flags
        _infer_epilogue_flags(v, attr)

        # optional: infer MNK (prefer out0)
        A = attr.shapes.get("in0", ())
        B = attr.shapes.get("in1", ())
        C = attr.shapes.get("out0", ())

        if len(A) == 2 and len(B) == 2 and len(C) == 2:
            M, N = int(C[0]), int(C[1])
            # K selection consistent with trans flags
            k_a = A[1] if (not transA) else A[0]
            k_b = B[0] if (not transB) else B[1]
            K = int(k_a) if int(k_a) == int(k_b) else int(k_a)
            attr.params.setdefault("mnk", (M, N, K))

        return attr
