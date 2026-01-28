# examples/python/aicf_fw/core_v2/kernel_select.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .ir import IRGraph


def _dtype_str(v) -> str:
    dt = getattr(v, "dtype", "torch.float32")
    return str(dt)


def _is_f16(s: str) -> bool:
    return ("float16" in s) or ("torch.float16" in s) or ("Half" in s)


def _is_f32(s: str) -> bool:
    return ("float32" in s) or ("torch.float32" in s) or ("Float" in s)


def _pick_kernel_id(op: str, in_dtypes: List[str], out_dtypes: List[str], attrs: Dict[str, Any]) -> Optional[str]:
    op = str(op).strip().lower()
    in0 = in_dtypes[0] if in_dtypes else ""
    out0 = out_dtypes[0] if out_dtypes else ""

    # ---- GemmEpilogue (NEW) ----
    if op == "gemm_epilogue":
        epi = str(attrs.get("epilogue", "bias_relu")).strip().lower()
        # 현재 registry에 등록된 건 bias+relu only
        if epi not in ("bias_relu", "bias+relu", "biasrelu"):
            return None

        if _is_f16(in0) or _is_f16(out0):
            return "gemm_bias_relu_f16_tc_wmma_out_f16_v0"
        return "gemm_bias_relu_f32_naive_v0"

    # ---- Gemm ----
    if op == "gemm":
        if _is_f16(in0) or _is_f16(out0):
            return "gemm_f16_tc_wmma_out_f16_v0"
        return "gemm_f32_naive_v0"

    # ---- BiasAdd ----
    if op == "bias_add":
        if _is_f16(in0) or _is_f16(out0):
            return "bias_add_f16_v0"
        return "bias_add_f32_v0"

    # ---- Add ----
    if op == "add":
        if _is_f16(in0) or _is_f16(out0):
            return "add_f16_v0"
        return "add_f32_v0"

    # ---- Relu ----
    if op == "relu":
        if _is_f16(in0) or _is_f16(out0):
            return "relu_f16_v0"
        return "relu_f32_v0"

    # ---- MseGrad ----
    if op == "mse_grad":
        if _is_f16(in0) or _is_f16(out0):
            return "mse_grad_f16_v0"
        return "mse_grad_f32_v0"

    # ---- ReluBwd ----
    if op == "relu_bwd":
        if _is_f16(in0) or _is_f16(out0):
            return "relu_bwd_f16_v0"
        return "relu_bwd_f32_v0"

    # ---- ReduceSum (keep_lastdim rowsum for bias grad) ----
    if op == "reduce_sum":
        if _is_f16(out0):
            return "reduce_sum_keep_lastdim_f16_v0"
        if _is_f16(in0):
            return "reduce_sum_keep_lastdim_f16_to_f32_v0"
        return "reduce_sum_keep_lastdim_f32_v0"

    # ---- SgdStep ----
    if op == "sgd_step":
        if _is_f16(in0) or _is_f16(out0):
            return "sgd_step_f16_v0"
        return "sgd_step_f32_v0"

    # ---- Copy ----
    if op in ("copy", "copy_saved", "copy_aux"):
        if _is_f16(in0) or _is_f16(out0):
            return "copy_f16_v0"
        return "copy_f32_v0"

    if op == "grad_zero":
        return "grad_zero_v0"
    if op == "adam_step":
        return "adam_step_f32_v0"
    if op == "step_inc":
        return "step_inc_v0"
    if op in ("bias_corr", "biascorr"):
        return "bias_corr_v0"

    if op == "layernorm_fwd":
        if _is_f16(in0) or _is_f16(out0):
            return "layernorm_fwd_f16_v0"
        return "layernorm_fwd_f32_v0"

    if op == "layernorm_bwd":
        if _is_f16(in0) or _is_f16(out0):
            return "layernorm_bwd_f16_v0"
        return "layernorm_bwd_f32_v0"

    if op == "batchnorm_fwd":
        return "batchnorm_fwd_f16_v0"
    if op == "batchnorm_bwd":
        return "batchnorm_bwd_f16_v0"

    return None


def apply_kernel_selection(ir: IRGraph, lowered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Stage B: attach kernel_id (pin decisions).
    """
    out: List[Dict[str, Any]] = []
    for it in lowered:
        op = str(it.get("op", "")).strip().lower()
        in_vids = [int(x) for x in it.get("inputs", [])]
        out_vids = [int(y) for y in it.get("outputs", [])]
        attrs = dict(it.get("attrs", {}) or {})

        in_dtypes = [_dtype_str(ir.values[int(v)]) for v in in_vids]
        out_dtypes = [_dtype_str(ir.values[int(v)]) for v in out_vids]

        kid = _pick_kernel_id(op, in_dtypes, out_dtypes, attrs)

        new_it = dict(it)
        if kid is not None:
            new_it["kernel_id"] = kid
        out.append(new_it)

    return out
