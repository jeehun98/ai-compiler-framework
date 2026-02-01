from __future__ import annotations
from typing import Dict, List

from .types import ExecPlan, LoweredOp
from ..builder import Builder


# 어떤 op가 "out0 can alias in0" 형태인지 정의
# (필요하면 add/relu 등도 여기에 추가)
_ALIAS_OUT0_IN0 = {
    "bias_add",
    "step_inc",
    "sgd_step",
    # "relu", "add" ... (추가 가능)
}


def make_exec_plan_cuda(b: Builder, lowered: List[LoweredOp]) -> ExecPlan:
    """
    Runtime execution plan decisions.

    For now:
      - slot alias for inplace_ok ops that support out0 alias in0.
      - special-case adam_step: (Pout,Mout,Vout) can alias (P,M,V)
    """
    alias: Dict[int, int] = {}

    # IR과 lowered는 같은 순서라고 가정 (lower_ir_cuda가 b.ops 순회)
    for op, lop in zip(b.ops, lowered):
        constraints = dict(getattr(op, "constraints", {}) or {})
        if not constraints:
            constraints = dict(getattr(lop, "constraints", {}) or {})
        inplace_ok = bool(constraints.get("inplace_ok", False))

        if not inplace_ok:
            continue

        # -----------------------------------------
        # adam_step: outputs 3개를 각각 입력 P/M/V에 alias
        # inputs : [P, G, M, V, bc1, bc2]
        # outputs: [Pout, Mout, Vout]
        # -----------------------------------------
        if op.kind == "adam_step":
            if len(op.inputs) == 6 and len(op.outputs) == 3:
                P, G, M, V, bc1, bc2 = op.inputs
                Pout, Mout, Vout = op.outputs

                # kernel constraint: Pout must NOT alias G
                if Pout == G:
                    raise ValueError("adam_step alias invalid: Pout cannot alias grad input G")

                alias[Pout] = P
                alias[Mout] = M
                alias[Vout] = V
            continue

        # -----------------------------------------
        # out0 -> in0 형태 alias
        # -----------------------------------------
        if op.kind in _ALIAS_OUT0_IN0:
            if len(op.inputs) >= 1 and len(op.outputs) == 1:
                out_vid = op.outputs[0]
                in0_vid = op.inputs[0]
                alias[out_vid] = in0_vid

        # (미래) 더 복잡한 alias 룰:
        # - add: out can alias one of inputs if refcount==1 등
        # - relu: out can alias in0 if no saved-for-bwd etc

    return ExecPlan(lowered=lowered, alias=alias)
