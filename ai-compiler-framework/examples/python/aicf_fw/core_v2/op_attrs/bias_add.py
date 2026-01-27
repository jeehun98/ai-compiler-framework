# examples/python/aicf_fw/core_v2/op_attrs/bias_add.py
from __future__ import annotations

from typing import Any, Optional, Tuple
from .base import OpAttr, LoweredOpView, ValueDescs, fill_common


def _infer_broadcast_axis(x_shape: Tuple[int, ...], b_shape: Tuple[int, ...]) -> Optional[int]:
    """
    Best-effort broadcast axis inference.
    Typical cases:
      x: (B, D), b: (D,)  -> axis = -1 (or 1)
      x: (B, S, D), b: (D,) -> axis = -1
      x: (B, S, D), b: (1, 1, D) -> axis = -1
    We return axis in Python-style (can be negative).
    """
    if not x_shape or not b_shape:
        return None

    # If bias is 1D and matches last dim -> axis=-1
    if len(b_shape) == 1 and len(x_shape) >= 1 and b_shape[0] == x_shape[-1]:
        return -1

    # If bias has same rank and matches only one dim with others 1 -> find that dim
    if len(b_shape) == len(x_shape):
        candidate = None
        for i, (xs, bs) in enumerate(zip(x_shape, b_shape)):
            if bs == xs:
                candidate = i
            elif bs == 1:
                continue
            else:
                return None  # incompatible pattern
        if candidate is not None:
            # return as negative axis (more canonical for "feature dim" use)
            return candidate - len(x_shape)
        return None

    # If bias rank < x rank, align to tail (numpy broadcast)
    if len(b_shape) < len(x_shape):
        # compare b_shape against last len(b_shape) dims
        tail = x_shape[-len(b_shape):]
        ok = True
        candidate = None
        for j, (xs, bs) in enumerate(zip(tail, b_shape)):
            if bs == xs:
                candidate = j  # within tail
            elif bs == 1:
                continue
            else:
                ok = False
                break
        if ok and candidate is not None:
            # convert tail index to full index then to negative
            full_i = (len(x_shape) - len(b_shape)) + candidate
            return full_i - len(x_shape)

    return None


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

        x_shape = attr.shapes.get("in0", ())
        bias_shape = attr.shapes.get("in1", ())
        out_shape = attr.shapes.get("out0", ())

        # 1) record bias shape
        attr.params.setdefault("bias_shape", bias_shape)

        # 2) in-place hint (your lowered often has out vid == in0 vid)
        if len(v.inputs) >= 1 and len(v.outputs) >= 1:
            attr.params.setdefault("inplace", (v.inputs[0] == v.outputs[0]))

        # 3) broadcast axis (best-effort)
        axis = _infer_broadcast_axis(x_shape, bias_shape)
        if axis is not None:
            attr.params.setdefault("broadcast_axis", int(axis))

        # 4) expected_out_shape (if out0 not available, fallback to x_shape)
        if out_shape:
            attr.params.setdefault("expected_out_shape", out_shape)
        elif x_shape:
            attr.params.setdefault("expected_out_shape", x_shape)

        return attr
