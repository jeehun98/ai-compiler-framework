# examples/python/aicf_fw/core_v2/op_attrs/gemm.py
from __future__ import annotations
from typing import Any, Optional, Tuple
from .base import OpAttr, LoweredOpView, ValueDescs, fill_common


def _pick_k_from_shapes(
    A: Tuple[int, ...],
    B: Tuple[int, ...],
    transA: bool,
    transB: bool,
) -> Optional[int]:
    """
    GEMM:  A @ B -> C
      If transA: A used as A^T
      If transB: B used as B^T

    We only need K. For 2D:
      A_used: (M, K) if not transA else (K, M)
      B_used: (K, N) if not transB else (N, K)
    So K is:
      - from A: A[1] if not transA else A[0]
      - from B: B[0] if not transB else B[1]
    Prefer consistent K if both available.
    """
    if len(A) != 2 or len(B) != 2:
        return None

    k_a = A[1] if (not transA) else A[0]
    k_b = B[0] if (not transB) else B[1]

    if k_a == k_b:
        return int(k_a)

    # mismatch: pick a sane one (prefer positive, non-zero)
    if int(k_a) > 0:
        return int(k_a)
    if int(k_b) > 0:
        return int(k_b)
    return None


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
        transA = bool(v.attrs.get("transA", v.attrs.get("trans_a", False)))
        transB = bool(v.attrs.get("transB", v.attrs.get("trans_b", False)))
        attr.layout["transA"] = transA
        attr.layout["transB"] = transB

        # infer MNK (best-effort, but robust)
        A = attr.shapes.get("in0", ())
        B = attr.shapes.get("in1", ())
        C = attr.shapes.get("out0", ())

        # M,N: prefer out0 (most reliable), else derive from A/B
        M: Optional[int] = None
        N: Optional[int] = None

        if len(C) == 2:
            M, N = int(C[0]), int(C[1])
        else:
            # fallback: derive M from A_used, N from B_used
            if len(A) == 2:
                M = int(A[0]) if (not transA) else int(A[1])
            if len(B) == 2:
                N = int(B[1]) if (not transB) else int(B[0])

        K = _pick_k_from_shapes(A, B, transA=transA, transB=transB)

        # store only if we got a full triple
        if M is not None and N is not None and K is not None:
            attr.params.setdefault("mnk", (int(M), int(N), int(K)))

        return attr
