# examples/python/aicf_fw/core_v2/lower.py
from __future__ import annotations

from typing import Any, Dict, List

from .ir import IRGraph


def _add_item(lowered: List[Dict[str, Any]], item: Dict[str, Any]) -> None:
    """
    Stage A: lowering only.
    - DO NOT attach kernel_id here.
    - Keep attrs for later opt/kernel-select passes.
    """
    # normalize a bit for safety
    item = dict(item)
    item["op"] = str(item["op"]).strip().lower()
    item["inputs"] = [int(x) for x in item.get("inputs", [])]
    item["outputs"] = [int(y) for y in item.get("outputs", [])]
    item["attrs"] = dict(item.get("attrs", {}) or {})
    lowered.append(item)


def lower_to_backend_ops(ir: IRGraph) -> List[Dict[str, Any]]:
    """
    IR(core_v2) -> backend lowered ops (primitive ops only).
    Produces ops compatible with exec.py (_OPNAME_TO_KIND).

    IMPORTANT:
    - This stage must be semantics-preserving.
    - No kernel selection / no kernel_id pinning here.
    - Kernel selection belongs to a later stage (e.g. kernel_select.py).
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

            _add_item(lowered, {"op": "gemm", "inputs": [x_vid, w_vid], "outputs": [y_vid], "attrs": {"transB": True}})

            if bool(attrs.get("bias", False)) and len(ins) >= 3:
                b_vid = ins[2]
                _add_item(lowered, {"op": "bias_add", "inputs": [y_vid, b_vid], "outputs": [y_vid], "attrs": {}})
            continue

        # ---- Save -> copy_saved ----
        if op == "save":
            _add_item(lowered, {"op": "copy_saved", "inputs": [ins[0]], "outputs": [outs[0]], "attrs": {}})
            continue

        # ---- ReLU ----
        if op == "relu":
            _add_item(lowered, {"op": "relu", "inputs": [ins[0]], "outputs": [outs[0]], "attrs": {}})
            continue

        # ---- MseGrad ----
        if op in ("msegrad", "mse_grad"):
            _add_item(lowered, {"op": "mse_grad", "inputs": [ins[0], ins[1]], "outputs": [outs[0]], "attrs": attrs})
            continue

        # ---- ReluBwd ----
        if op in ("relubwd", "relu_bwd"):
            _add_item(lowered, {"op": "relu_bwd", "inputs": [ins[0], ins[1]], "outputs": [outs[0]], "attrs": {}})
            continue

        # ---- LinearBwd -> gemm(dx) + gemm(dW) + reduce_sum(db) ----
        if op in ("linearbwd", "linear_bwd"):
            # IR LinearBwd inputs: (x, W, dY)
            # outputs: (dx, dW, db?)  with bias flag
            x_vid, w_vid, dy_vid = ins[0], ins[1], ins[2]
            dx_vid = outs[0]
            dW_vid = outs[1]

            # dx = dY @ W  (transA=False, transB=False)
            _add_item(
                lowered,
                {"op": "gemm", "inputs": [dy_vid, w_vid], "outputs": [dx_vid], "attrs": {"transA": False, "transB": False}},
            )

            # dW = dY^T @ x (transA=True, transB=False)
            _add_item(
                lowered,
                {"op": "gemm", "inputs": [dy_vid, x_vid], "outputs": [dW_vid], "attrs": {"transA": True, "transB": False}},
            )

            if bool(attrs.get("bias", False)) and len(outs) >= 3:
                db_vid = outs[2]
                # keep_lastdim rowsum (bias grad); still expressed as reduce_sum primitive here
                _add_item(lowered, {"op": "reduce_sum", "inputs": [dy_vid], "outputs": [db_vid], "attrs": {"axis": 0, "keepdim": False}})
            continue

        # ---- SgdStep ----
        if op in ("sgdstep", "sgd_step"):
            # IR SgdStep inputs: (p, g) outputs: (p)
            _add_item(lowered, {"op": "sgd_step", "inputs": [ins[0], ins[1]], "outputs": [outs[0]], "attrs": attrs})
            continue

        # ---- Already-lowered primitives (keep them working) ----
        primitive = op
        if primitive in (
            "gemm",
            "bias_add",
            "add",
            "relu",
            "reduce_sum",
            "mse_grad",
            "relu_bwd",
            "copy",
            "copy_saved",
            "copy_aux",
            "grad_zero",
            "adam_step",
            "step_inc",
            "bias_corr",
            "layernorm_fwd",
            "layernorm_bwd",
            "batchnorm_fwd",
            "batchnorm_bwd",
            "sgd_step",
        ):
            _add_item(lowered, {"op": primitive, "inputs": ins, "outputs": outs, "attrs": attrs})
            continue

        raise RuntimeError(f"[lower] unsupported IR op '{ir_op}' (normalized='{op}')")

    return lowered
