# examples/python/aicf_fw/core_v2/lower.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .ir import IRGraph


def _dtype_str(v) -> str:
    dt = getattr(v, "dtype", "torch.float32")
    return str(dt)


def _pick_kernel_id(op: str, in_dtypes: List[str], out_dtypes: List[str], attrs: Dict[str, Any]) -> Optional[str]:
    op = str(op).strip().lower()
    in0 = in_dtypes[0] if in_dtypes else ""
    out0 = out_dtypes[0] if out_dtypes else ""

    def is_f16(s: str) -> bool:
        return ("float16" in s) or ("torch.float16" in s) or ("Half" in s)

    def is_f32(s: str) -> bool:
        return ("float32" in s) or ("torch.float32" in s) or ("Float" in s)

    if op == "gemm":
        if is_f16(in0) or is_f16(out0):
            return "gemm_f16_tc_wmma_out_f16_v0"
        return "gemm_f32_naive_v0"

    if op == "bias_add":
        if is_f16(in0) or is_f16(out0):
            return "bias_add_f16_v0"
        return "bias_add_f32_v0"

    if op == "add":
        if is_f16(in0) or is_f16(out0):
            return "add_f16_v0"
        return "add_f32_v0"

    if op == "relu":
        if is_f16(in0) or is_f16(out0):
            return "relu_f16_v0"
        return "relu_f32_v0"

    if op == "mse_grad":
        if is_f16(in0) or is_f16(out0):
            return "mse_grad_f16_v0"
        return "mse_grad_f32_v0"

    if op == "relu_bwd":
        if is_f16(in0) or is_f16(out0):
            return "relu_bwd_f16_v0"
        return "relu_bwd_f32_v0"

    if op == "reduce_sum":
        # 결정 박제 기준:
        # reduce_sum_keep_lastdim = bias grad (rowsum over leading dims)
        # - out f16  => reduce_sum_keep_lastdim_f16_v0
        # - out f32  => (in f16) reduce_sum_keep_lastdim_f16_to_f32_v0
        #            => (in f32) reduce_sum_keep_lastdim_f32_v0

        if is_f16(out0):
            return "reduce_sum_keep_lastdim_f16_v0"

        if is_f16(in0):
            return "reduce_sum_keep_lastdim_f16_to_f32_v0"
        return "reduce_sum_keep_lastdim_f32_v0"


    if op == "sgd_step":
        if is_f16(in0) or is_f16(out0):
            return "sgd_step_f16_v0"
        return "sgd_step_f32_v0"

    if op in ("copy", "copy_saved", "copy_aux"):
        if is_f16(in0) or is_f16(out0):
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
        if is_f16(in0) or is_f16(out0):
            return "layernorm_fwd_f16_v0"
        return "layernorm_fwd_f32_v0"

    if op == "layernorm_bwd":
        if is_f16(in0) or is_f16(out0):
            return "layernorm_bwd_f16_v0"
        return "layernorm_bwd_f32_v0"

    if op == "batchnorm_fwd":
        return "batchnorm_fwd_f16_v0"

    if op == "batchnorm_bwd":
        return "batchnorm_bwd_f16_v0"

    return None


def _add_item(lowered: List[Dict[str, Any]], ir: IRGraph, item: Dict[str, Any]) -> None:
    op = str(item["op"]).strip().lower()
    in_vids = [int(x) for x in item.get("inputs", [])]
    out_vids = [int(y) for y in item.get("outputs", [])]
    attrs = dict(item.get("attrs", {}) or {})

    in_dtypes = [_dtype_str(ir.values[int(v)]) for v in in_vids]
    out_dtypes = [_dtype_str(ir.values[int(v)]) for v in out_vids]

    kid = _pick_kernel_id(op, in_dtypes, out_dtypes, attrs)
    if kid is not None:
        item["kernel_id"] = kid
    lowered.append(item)


def lower_to_backend_ops(ir: IRGraph) -> List[Dict[str, Any]]:
    """
    IR(core_v2) -> backend lowered ops (primitive ops only).
    Produces ops compatible with exec.py (_OPNAME_TO_KIND) AND attaches kernel_id (Stage A).
    """
    lowered: List[Dict[str, Any]] = []

    for n in ir.nodes:
        ir_op = str(getattr(n, "op", getattr(n, "kind", ""))).strip()
        op = ir_op.lower()


        # ---- aliases / normalize (CamelCase -> snake_case primitive names) ----
        if op == "adamstep":
            op = "adam_step"
        elif op == "stepinc":
            op = "step_inc"
        elif op == "biascorr":
            op = "bias_corr"
            
        ins = [int(x) for x in getattr(n, "inputs", [])]
        outs = [int(y) for y in getattr(n, "outputs", [])]
        attrs = dict(getattr(n, "attrs", {}) or {})



        # ---- Linear -> gemm + bias_add ----
        if op == "linear":
            # IR Linear inputs: (x, W, b?) outputs: (y)
            # backend layout: gemm(x, W)->y with transB=True then bias_add(y,b)->y
            x_vid = ins[0]
            w_vid = ins[1]
            y_vid = outs[0]

            _add_item(lowered, ir, {"op": "gemm", "inputs": [x_vid, w_vid], "outputs": [y_vid], "attrs": {"transB": True}})
            if bool(attrs.get("bias", False)) and len(ins) >= 3:
                b_vid = ins[2]
                _add_item(lowered, ir, {"op": "bias_add", "inputs": [y_vid, b_vid], "outputs": [y_vid], "attrs": {}})
            continue

        # ---- Save -> copy_saved ----
        if op == "save":
            _add_item(lowered, ir, {"op": "copy_saved", "inputs": [ins[0]], "outputs": [outs[0]], "attrs": {}})
            continue

        # ---- ReLU ----
        if op == "relu":
            _add_item(lowered, ir, {"op": "relu", "inputs": [ins[0]], "outputs": [outs[0]], "attrs": {}})
            continue

        # ---- MseGrad ----
        if op == "msegrad" or op == "mse_grad":
            _add_item(lowered, ir, {"op": "mse_grad", "inputs": [ins[0], ins[1]], "outputs": [outs[0]], "attrs": attrs})
            continue

        # ---- ReluBwd ----
        if op == "relubwd" or op == "relu_bwd":
            _add_item(lowered, ir, {"op": "relu_bwd", "inputs": [ins[0], ins[1]], "outputs": [outs[0]], "attrs": {}})
            continue

        # ---- LinearBwd -> gemm(dx) + gemm(dW) + reduce_sum(db) ----
        if op == "linearbwd" or op == "linear_bwd":
            # IR LinearBwd inputs: (x, W, dY)
            # outputs: (dx, dW, db?)  with bias flag
            x_vid, w_vid, dy_vid = ins[0], ins[1], ins[2]
            dx_vid = outs[0]
            dW_vid = outs[1]

            # dx = dY @ W  (transA=False, transB=False)
            _add_item(lowered, ir, {"op": "gemm", "inputs": [dy_vid, w_vid], "outputs": [dx_vid], "attrs": {"transA": False, "transB": False}})

            # dW = dY^T @ x (transA=True, transB=False)
            _add_item(lowered, ir, {"op": "gemm", "inputs": [dy_vid, x_vid], "outputs": [dW_vid], "attrs": {"transA": True, "transB": False}})

            if bool(attrs.get("bias", False)) and len(outs) >= 3:
                db_vid = outs[2]
                _add_item(lowered, ir, {"op": "reduce_sum", "inputs": [dy_vid], "outputs": [db_vid], "attrs": {"axis": 0, "keepdim": False}})
            continue

        # ---- SgdStep ----
        if op == "sgdstep" or op == "sgd_step":
            # IR SgdStep inputs: (p, g) outputs: (p)
            _add_item(lowered, ir, {"op": "sgd_step", "inputs": [ins[0], ins[1]], "outputs": [outs[0]], "attrs": attrs})
            continue

        # ---- Already-lowered primitives (just normalize name) ----
        # If you ever emit primitives directly, keep them working:
        primitive = op
        if primitive in ("gemm", "bias_add", "add", "relu", "reduce_sum", "mse_grad", "relu_bwd", "copy", "copy_saved", "copy_aux", "grad_zero", "adam_step", "step_inc", "bias_corr", "layernorm_fwd", "layernorm_bwd", "batchnorm_fwd", "batchnorm_bwd", "sgd_step"):
            _add_item(lowered, ir, {"op": primitive, "inputs": ins, "outputs": outs, "attrs": attrs})
            continue

        raise RuntimeError(f"[lower] unsupported IR op '{ir_op}' (normalized='{op}')")

    return lowered
