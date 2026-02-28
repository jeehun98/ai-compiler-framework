# python/aicf_v2/src/aicf_v2/emitters/cuda/relu.py
from __future__ import annotations

from typing import Any, Dict

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved, OpFlags


def emit(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    x: int,
    out: int,
    name: str = "relu",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """ReLU Forward: y = max(x, 0)"""

    in_role = ["x"]
    out_role = ["y"]

    static = OpFlags.IS_ELEMENTWISE | OpFlags.IS_ACTIVATION

    return emit_resolved(
        b,
        kind="relu",
        name=name,
        inputs=[x],
        outputs=[out],
        kind_id=ctx.EltwiseRelu,
        attr_schema=0,
        attr_blob=b"",
        attrs={
            "in_role": in_role,
            "out_role": out_role,
        },
        constraints=constraints,
        hints=hints,
        static_flags=static,
    )


def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,
    grad_y: int,
    name: str = "relu_bwd",
) -> Dict[int, int]:
    """
    Standard ReLU backward using fwd output y:
      dx = dy * (y > 0)

    Backend contract:
      kind_id = ctx.ReluBwd
      inputs  = [dy, y]
      outputs = [dx]
    """
    y_vid = fwd_node.outputs[0]
    x_vid = fwd_node.inputs[0]

    dx_vid = b.value(f"{name}.dx", b.values[x_vid].spec)

    in_role = ["grad_y", "y"]
    out_role = ["grad_x"]

    static = OpFlags.IS_ELEMENTWISE | OpFlags.IS_ACTIVATION

    emit_resolved(
        b,
        kind="relu_bwd",
        name=name,
        inputs=[grad_y, y_vid],
        outputs=[dx_vid],
        kind_id=ctx.ReluBwd,
        attr_schema=0,
        attr_blob=b"",
        attrs={
            "in_role": in_role,
            "out_role": out_role,
        },
        constraints={"inplace_ok": True},
        hints=None,
        static_flags=static,
    )

    return {x_vid: dx_vid}


def emit_mask_from_y(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    y: int,
    grad_y: int,
    out: int,
    name: str = "relu_mask",
) -> int:
    """
    Helper for fused epilogues:
      out = grad_y * (y > 0)

    Uses the same backend opkind as relu_bwd:
      kind_id = ctx.ReluBwd
      inputs  = [grad_y, y]
      outputs = [out]
    """
    in_role = ["grad_y", "y"]
    out_role = ["grad_x"]

    static = OpFlags.IS_ELEMENTWISE | OpFlags.IS_ACTIVATION

    return emit_resolved(
        b,
        kind="relu_bwd",
        name=name,
        inputs=[grad_y, y],
        outputs=[out],
        kind_id=ctx.ReluBwd,
        attr_schema=0,
        attr_blob=b"",
        attrs={
            "in_role": in_role,
            "out_role": out_role,
        },
        constraints={"inplace_ok": True},
        hints=None,
        static_flags=static,
    )