# aicf_fw/core_v2/lowering.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .ir import IRGraph


# lowered op dict schema:
# {
#   "op": "gemm" | "relu" | "bias_add" | "mse_grad" | "copy" | "relu_bwd" | "reduce_sum" | ...
#   "inputs": [vid...],
#   "outputs": [vid...],
#   "attrs": {...}
# }


@dataclass
class LoweringOptions:
    """
    Stage2/5A 지원 op:
    - Linear: gemm + (optional) bias_add
    - ReLU: relu
    - MseGrad: mse_grad
    - Save: copy
    - ReluBwd: relu_bwd
    - LinearBwd: gemm + gemm + (optional) reduce_sum
    """
    pass


def lower_to_backend_ops(ir: IRGraph, *, opts: Optional[LoweringOptions] = None) -> List[Dict[str, Any]]:
    if opts is None:
        opts = LoweringOptions()

    lowered: List[Dict[str, Any]] = []

    def emit(op: str, inputs: List[int], outputs: List[int], attrs: Optional[Dict[str, Any]] = None):
        lowered.append(
            {
                "op": op,
                "inputs": [int(x) for x in inputs],
                "outputs": [int(y) for y in outputs],
                "attrs": dict(attrs or {}),
            }
        )

    for n in ir.nodes:
        # ------------------------------------------------------------
        # Forward
        # ------------------------------------------------------------
        if n.op == "Linear":
            # IR: Linear(x, W, b?) -> y
            # Lowered:
            #   gemm(x, W) -> y   (transB=True)
            #   bias_add(y, b) -> y   (in-place)
            if len(n.inputs) not in (2, 3):
                raise RuntimeError(f"lower(Linear): expected 2 or 3 inputs, got {len(n.inputs)}: {n.inputs}")
            if len(n.outputs) != 1:
                raise RuntimeError(f"lower(Linear): expected 1 output, got {len(n.outputs)}: {n.outputs}")

            x_vid = int(n.inputs[0])
            W_vid = int(n.inputs[1])
            y_vid = int(n.outputs[0])

            emit("gemm", [x_vid, W_vid], [y_vid], {"transB": True})

            has_bias = bool(n.attrs.get("bias", False)) and len(n.inputs) == 3
            if has_bias:
                b_vid = int(n.inputs[2])
                emit("bias_add", [y_vid, b_vid], [y_vid], {})

            continue

        if n.op == "ReLU":
            # IR: ReLU(x)->y
            if len(n.inputs) != 1 or len(n.outputs) != 1:
                raise RuntimeError(f"lower(ReLU): expected 1 in/1 out, got in={n.inputs}, out={n.outputs}")
            emit("relu", [int(n.inputs[0])], [int(n.outputs[0])], {})
            continue

        if n.op == "MseGrad":
            # IR: MseGrad(pred, target)->dY
            if len(n.inputs) != 2 or len(n.outputs) != 1:
                raise RuntimeError(f"lower(MseGrad): expected 2 in/1 out, got in={n.inputs}, out={n.outputs}")
            attrs: Dict[str, Any] = {}
            if "scale" in n.attrs:
                attrs["scale"] = float(n.attrs["scale"])
            emit("mse_grad", [int(n.inputs[0]), int(n.inputs[1])], [int(n.outputs[0])], attrs)
            continue

        # ------------------------------------------------------------
        # Backward (Stage5A)
        # ------------------------------------------------------------
        if n.op == "Save":
            # IR: Save(x) -> saved
            # Lowered: copy(x) -> saved
            if len(n.inputs) != 1 or len(n.outputs) != 1:
                raise RuntimeError(f"lower(Save): expected 1 in/1 out, got in={n.inputs}, out={n.outputs}")
            emit("copy", [int(n.inputs[0])], [int(n.outputs[0])], {})
            continue

        if n.op == "ReluBwd":
            # IR: ReluBwd(dout, saved_y) -> din
            # Lowered: relu_bwd(dout, saved_y) -> din
            if len(n.inputs) != 2 or len(n.outputs) != 1:
                raise RuntimeError(f"lower(ReluBwd): expected 2 in/1 out, got in={n.inputs}, out={n.outputs}")
            emit("relu_bwd", [int(n.inputs[0]), int(n.inputs[1])], [int(n.outputs[0])], {})
            continue

        if n.op == "LinearBwd":
            # IR: LinearBwd(x, W, dY) -> (dX, dW, (db?))
            # y = x @ W^T (+b)
            #
            # dX = dY @ W
            # dW = dY^T @ x
            # db = reduce_sum(dY, axis=0)
            if len(n.inputs) != 3:
                raise RuntimeError(f"lower(LinearBwd): expected 3 inputs, got {len(n.inputs)}: {n.inputs}")
            if len(n.outputs) not in (2, 3):
                raise RuntimeError(f"lower(LinearBwd): expected 2 or 3 outputs, got {len(n.outputs)}: {n.outputs}")

            x_vid = int(n.inputs[0])
            W_vid = int(n.inputs[1])
            dY_vid = int(n.inputs[2])

            dX_vid = int(n.outputs[0])
            dW_vid = int(n.outputs[1])

            # dX = dY @ W
            emit("gemm", [dY_vid, W_vid], [dX_vid], {"transA": False, "transB": False})

            # dW = dY^T @ x  (shape OUT x IN)
            emit("gemm", [dY_vid, x_vid], [dW_vid], {"transA": True, "transB": False})

            # db = reduce_sum(dY, axis=0)
            if len(n.outputs) == 3:
                db_vid = int(n.outputs[2])
                emit("reduce_sum", [dY_vid], [db_vid], {"axis": 0, "keepdim": False})

            continue

        # Stage2/5A에서는 나머지는 명시적으로 실패 (확장 시점에 추가)
        raise RuntimeError(f"lower: unsupported IR op '{n.op}' in stage2/5A")

    return lowered
