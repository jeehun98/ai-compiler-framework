from __future__ import annotations

from ..types import ExecPlan
from ...builder import Builder


def optimize_ir(b: Builder) -> Builder:
    """
    Future hook:
      - fuse passes (gemm+bias_add -> gemm_epilogue)
      - DCE, CSE, constant folding
      - layout/shape canonicalization
    For now: identity.
    """
    return b


def optimize_plan(plan: ExecPlan) -> ExecPlan:
    """
    Future hook:
      - schedule reordering (if legal)
      - kernel variant selection
      - workspace planning
      - cuda graph capture decision
    For now: identity.
    """
    return plan
