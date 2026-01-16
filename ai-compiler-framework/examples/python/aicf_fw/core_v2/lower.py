# aicf_fw/core_v2/lower.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .ir import IRGraph


@dataclass
class LoweringOptions:
    pass


def lower_to_backend_ops(ir: IRGraph, *, opts: LoweringOptions | None = None) -> List[Dict[str, Any]]:
    if opts is None:
        opts = LoweringOptions()

    lowered: List[Dict[str, Any]] = []

    def emit(op: str, inputs: List[int], outputs: List[int], attrs: Dict[str, Any] | None = None):
        lowered.append(
            {
                "op": str(op),
                "inputs": [int(x) for x in inputs],
                "outputs": [int(y) for y in outputs],
                "attrs": dict(attrs or {}),
            }
        )

    for n in ir.nodes:
        op = n.op

        # -------------------------
        # Forward
        # -------------------------
        if op == "Linear":
            if len(n.inputs) not in (2, 3):
                raise RuntimeError(f"lower(Linear): expected 2 or 3 inputs, got {len(n.inputs)}")
            if len(n.outputs) != 1:
                raise RuntimeError("lower(Linear): expected 1 output")

            x_vid = int(n.inputs[0])
            W_vid = int(n.inputs[1])
            y_vid = int(n.outputs[0])

            emit("gemm", [x_vid, W_vid], [y_vid], {"transB": True})

            # FIX: bias 존재 판정
            has_bias = bool(n.attrs.get("bias", False)) or (len(n.inputs) == 3)
            if has_bias:
                if len(n.inputs) != 3:
                    raise RuntimeError("lower(Linear): bias requested but no bias input")
                b_vid = int(n.inputs[2])
                emit("bias_add", [y_vid, b_vid], [y_vid], {})
            continue

        if op == "ReLU":
            if len(n.inputs) != 1 or len(n.outputs) != 1:
                raise RuntimeError("lower(ReLU): expected 1 in/1 out")
            emit("relu", [int(n.inputs[0])], [int(n.outputs[0])], {})
            continue

        if op == "Save":
            if len(n.inputs) != 1 or len(n.outputs) != 1:
                raise RuntimeError("lower(Save): expected 1 in/1 out")
            emit("copy", [int(n.inputs[0])], [int(n.outputs[0])], {})
            continue

        if op == "MseGrad":
            if len(n.inputs) != 2 or len(n.outputs) != 1:
                raise RuntimeError("lower(MseGrad): expected 2 in/1 out")
            attrs: Dict[str, Any] = {}
            if "scale" in n.attrs:
                attrs["scale"] = float(n.attrs["scale"])
            emit("mse_grad", [int(n.inputs[0]), int(n.inputs[1])], [int(n.outputs[0])], attrs)
            continue

        # -------------------------
        # Backward
        # -------------------------
        if op == "LinearBwd":
            if len(n.inputs) != 3:
                raise RuntimeError("lower(LinearBwd): expected 3 inputs (x,W,dY)")
            if len(n.outputs) not in (2, 3):
                raise RuntimeError("lower(LinearBwd): expected 2 or 3 outputs (dX,dW,dB?)")

            X_vid  = int(n.inputs[0])
            W_vid  = int(n.inputs[1])
            dY_vid = int(n.inputs[2])

            dX_vid = int(n.outputs[0])
            dW_vid = int(n.outputs[1])
            db_vid = int(n.outputs[2]) if len(n.outputs) == 3 else None

            # dX = dY @ W
            emit("gemm", [dY_vid, W_vid], [dX_vid], {"transA": False, "transB": False})

            # dW = dY^T @ X
            emit("gemm", [dY_vid, X_vid], [dW_vid], {"transA": True, "transB": False})

            if db_vid is not None:
                emit("reduce_sum", [dY_vid], [db_vid], {"axis": 0, "keepdim": False})

            continue

        if op == "ReluBwd":
            if len(n.inputs) != 2 or len(n.outputs) != 1:
                raise RuntimeError("lower(ReluBwd): expected 2 in/1 out")
            emit("relu_bwd", [int(n.inputs[0]), int(n.inputs[1])], [int(n.outputs[0])], {})
            continue

        # -------------------------
        # Optimizer (Stage6)
        # -------------------------
        if op == "StepInc":
            if len(n.inputs) != 1 or len(n.outputs) != 1:
                raise RuntimeError("lower(StepInc): expected 1 in/1 out")
            emit("step_inc", [int(n.inputs[0])], [int(n.outputs[0])], {})
            continue

        if op == "BiasCorr":
            if len(n.inputs) != 1 or len(n.outputs) != 2:
                raise RuntimeError("lower(BiasCorr): expected 1 in/2 out")
            emit(
                "bias_corr",
                [int(n.inputs[0])],
                [int(n.outputs[0]), int(n.outputs[1])],
                {"beta1": float(n.attrs["beta1"]), "beta2": float(n.attrs["beta2"])},
            )
            continue

        if op == "AdamStep":
            if len(n.inputs) != 6 or len(n.outputs) != 3:
                raise RuntimeError("lower(AdamStep): expected 6 in/3 out")
            emit(
                "adam_step",
                [int(x) for x in n.inputs],
                [int(y) for y in n.outputs],
                {
                    "lr": float(n.attrs["lr"]),
                    "beta1": float(n.attrs["beta1"]),
                    "beta2": float(n.attrs["beta2"]),
                    "eps": float(n.attrs["eps"]),
                },
            )
            continue

        raise RuntimeError(f"lower: unsupported IR op '{op}'")

    return lowered
