from __future__ import annotations
from typing import Dict, List

from .types import ExecPlan, LoweredOp
from ..builder import Builder


def make_exec_plan_cuda(b: Builder, lowered: List[LoweredOp]) -> ExecPlan:
    """
    Runtime execution plan decisions.
    For now: only slot alias for inplace_ok ops.
    """
    alias: Dict[int, int] = {}

    # IR과 lowered는 같은 순서라고 가정 (lower_ir_cuda가 b.ops 순회)
    for op, lop in zip(b.ops, lowered):
        # bias_add(out) can alias input0 if inplace_ok
        if op.kind == "bias_add" and (getattr(op, "constraints", {}) or {}).get("inplace_ok", False):
            if len(op.inputs) >= 1 and len(op.outputs) == 1:
                out_vid = op.outputs[0]
                in0_vid = op.inputs[0]
                alias[out_vid] = in0_vid

        # (미래) relu inplace 가능 조건, add inplace 가능 조건 등 추가 가능

    return ExecPlan(lowered=lowered, alias=alias)
