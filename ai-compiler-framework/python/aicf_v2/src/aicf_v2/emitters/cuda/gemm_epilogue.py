# python/aicf_v2/src/aicf_v2/emitters/cuda/gemm_epilogue.py
from __future__ import annotations

import struct
from typing import Any, Dict, Sequence

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved, OpFlags


def _role_index(role_list: Sequence[str] | None, role: str) -> int:
    if not role_list:
        raise ValueError(f"missing role list while looking for role='{role}'")
    try:
        return list(role_list).index(role)
    except ValueError as e:
        raise ValueError(f"role '{role}' not found in roles={list(role_list)}") from e


def emit(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    A: int,
    B: int,
    bias: int,
    out: int,
    transA: bool = False,
    transB: bool = False,
    relu: bool = True,
    name: str = "gemm_epilogue",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """
    GEMM + Bias (+ optional ReLU) fused epilogue forward.

    AttrBlob (schema=ctx.SCHEMA_GMEP or 0):
      struct.pack("<iii", transA, transB, relu)
    C++ parses schema 0 or 'GPEL'(0x4750454C).
    """

    # ---- Role 계약 ----
    in_role = ["A", "B", "bias"]
    out_role = ["out"]

    ta = 1 if bool(transA) else 0
    tb = 1 if bool(transB) else 0
    r = 1 if bool(relu) else 0
    blob = struct.pack("<iii", ta, tb, r)

    # ---- Static Flags ----
    static = OpFlags.IS_GEMM_LIKE | OpFlags.HAS_BIAS
    if relu:
        static |= OpFlags.IS_ACTIVATION

    return emit_resolved(
        b,
        kind="gemm_epilogue",
        name=name,
        inputs=[A, B, bias],
        outputs=[out],
        kind_id=ctx.GemmEpilogue,
        attr_schema=getattr(ctx, "SCHEMA_GMEP", 0),
        attr_blob=blob,
        attrs={
            "transA": bool(transA),
            "transB": bool(transB),
            "relu": bool(relu),
            "in_role": in_role,
            "out_role": out_role,
        },
        constraints=constraints,
        hints=hints,
        static_flags=static,
    )


def _emit_dbias_relu_mask_f32(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    dY: int,
    Y: int,
    dBias_out: int,
    relu_enable: bool,
    name: str,
) -> int:
    """
    GemmEpilogueBwd variant: dBias = sum(dY_masked) with optional mask from Y.
    C++ check requires: dY and Y are rank2 f32, same strides, and dBias is rank1 f32.

    Backend:
      OpKind::GemmEpilogueBwd
      inputs:  [dY, Y]
      outputs: [dBias]
      attr:    uses GemmEpilogueAttrV0; only relu bit is used by kernel.
    """
    # ta/tb are ignored by this bwd kernel; keep them 0.
    blob = struct.pack("<iii", 0, 0, 1 if relu_enable else 0)

    in_role = ["dY", "Y"]
    out_role = ["dBias"]

    static = OpFlags.IS_REDUCE | OpFlags.HAS_BIAS

    return emit_resolved(
        b,
        kind="gemm_epilogue_bwd",  # 문자열은 디스패치에 중요하지 않음(kind_id가 핵심)
        name=name,
        inputs=[dY, Y],
        outputs=[dBias_out],
        kind_id=ctx.GemmEpilogueBwd,               # ★ register_all.cpp에 등록된 OpKind
        attr_schema=getattr(ctx, "SCHEMA_GMEP", 0),# ★ C++: 0 또는 'GPEL'
        attr_blob=blob,
        attrs={
            "relu": bool(relu_enable),
            "in_role": in_role,
            "out_role": out_role,
        },
        constraints=None,
        hints=None,
        static_flags=static,
    )


def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,
    grad_y: int,
    name: str = "gemm_epilogue_bwd",
) -> Dict[int, int]:
    """
    Backward for gemm_epilogue (composed):

      if relu:
        dZ = dY * (Y > 0)   using ctx.ReluBwd (inputs=[dY, Y] -> dZ)
      else:
        dZ = dY

      dA/dB: reuse gemm.emit_bwd() with dZ
      dBias:
        - if dtype is f32 and ctx.GemmEpilogueBwd exists, use GemmEpilogueBwd kernel variant
          (expects dY and Y have identical strides)
        - else fallback to reduce_sum(dZ, axis=0)

    Returns:
      {A_vid: dA_vid, B_vid: dB_vid, bias_vid: dBias_vid}
    """
    # Local imports to avoid circular deps
    from .gemm import emit_bwd as emit_gemm_bwd
    from .reduce_sum import emit as emit_reduce_sum
    from .relu import emit_mask_from_y as emit_relu_mask_from_y

    in_role = fwd_node.attrs.get("in_role", ["A", "B", "bias"])
    A_vid = fwd_node.inputs[_role_index(in_role, "A")]
    B_vid = fwd_node.inputs[_role_index(in_role, "B")]
    bias_vid = fwd_node.inputs[_role_index(in_role, "bias")]
    Y_vid = fwd_node.outputs[0]

    ta = bool(fwd_node.attrs.get("transA", False))
    tb = bool(fwd_node.attrs.get("transB", False))
    relu = bool(fwd_node.attrs.get("relu", True))

    # 1) dZ (must match Y layout for GemmEpilogueBwd check)
    dZ_vid = grad_y
    if relu:
        dZ_vid = b.value(f"{name}.dZ", b.values[Y_vid].spec)
        emit_relu_mask_from_y(
            b,
            ctx,
            y=Y_vid,
            grad_y=grad_y,
            out=dZ_vid,
            name=f"{name}.relu_mask",
        )

    # 2) dA/dB using existing GEMM bwd
    fake_gemm_node = type("TmpGemmNode", (), {})()
    fake_gemm_node.inputs = [A_vid, B_vid]
    fake_gemm_node.outputs = [Y_vid]
    fake_gemm_node.attrs = {"transA": ta, "transB": tb, "in_role": ["A", "B"]}

    grads_ab = emit_gemm_bwd(b, ctx, fake_gemm_node, dZ_vid, name=f"{name}.gemm")

    # 3) dBias
    dBias_vid = b.value(f"{name}.dBias", b.values[bias_vid].spec)

    can_use_dbias_kernel = hasattr(ctx, "GemmEpilogueBwd") and (b.values[Y_vid].spec.dtype == "f32")

    if can_use_dbias_kernel:
        # We already applied mask in dZ when relu=True, so set relu_enable=False here.
        # This avoids double-masking and matches semantics.
        _emit_dbias_relu_mask_f32(
            b,
            ctx,
            dY=dZ_vid,
            Y=Y_vid,
            dBias_out=dBias_vid,
            relu_enable=False,
            name=f"{name}.dbias",
        )
    else:
        # Generic fallback (works for f16 too if reduce_sum supports it)
        emit_reduce_sum(b, ctx, x=dZ_vid, out=dBias_vid, axis=0)

    grads_ab[bias_vid] = dBias_vid
    return grads_ab