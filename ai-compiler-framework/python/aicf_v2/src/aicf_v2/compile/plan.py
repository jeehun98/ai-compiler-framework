from __future__ import annotations
from typing import Dict

from .types import ExecPlan
from ..builder import Builder


# 어떤 op가 "out0 can alias in0" 형태인지 정의
_ALIAS_OUT0_IN0 = {
    "bias_add",
    "step_inc",
    "sgd_step",
    # "softmax",
    # 필요하면 추가: "copy", "grad_zero" 등은 의미가 다르니 정책 확정 후
}


def make_exec_plan_cuda(b: Builder) -> ExecPlan:
    """
    Runtime execution plan decisions.

    For now:
      - slot alias for inplace_ok ops that support out0 alias in0.
      - special-case adam_step: (Pout,Mout,Vout) can alias (P,M,V)

    Note:
      - lower 단계가 없으므로, alias 판단의 기준은 b.ops의 constraints/hints만 사용.
    """
    alias: Dict[int, int] = {}

    for op in b.ops:
        constraints = dict(getattr(op, "constraints", {}) or {})
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

    return ExecPlan(ops=list(b.ops), alias=alias)
