# examples/python/aicf_fw/core_v2/op_attrs/registry.py
from __future__ import annotations
from typing import Any, Dict, Optional
from .base import OpAttr, LoweredOpView, ValueDescs, fill_common
from .gemm import GemmAttrBuilder
from .bias_add import BiasAddAttrBuilder
# 필요하면 relu/reduce_sum/... 계속 추가

class DefaultAttrBuilder:
    kind = "*"
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
        attr.sig = v.kind.upper()
        fill_common(attr, v, value_descs)
        return attr


_BUILDERS: Dict[str, Any] = {
    "gemm": GemmAttrBuilder(),
    "bias_add": BiasAddAttrBuilder(),
    # "relu": ReluAttrBuilder(),
    # "reduce_sum": ReduceSumAttrBuilder(),
}

_DEFAULT = DefaultAttrBuilder()

def build_op_attr(op: Any, value_descs: ValueDescs, op_id: Optional[int] = None) -> OpAttr:
    v = LoweredOpView.from_any(op, op_id=op_id)
    b = _BUILDERS.get(v.kind.lower(), _DEFAULT)
    return b.build(op, value_descs, op_id=op_id)
