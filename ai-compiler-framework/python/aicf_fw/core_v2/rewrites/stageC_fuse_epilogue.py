from __future__ import annotations
from typing import Any, Dict, List


def stageC_fuse_gemm_epilogue(ir, lowered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fuse pattern (linear schedule, adjacent):
      gemm -> (y)
      bias_add (y, b) -> (y)   [optional, must be in-place on y]
      relu (y) -> (z)

    Rewrite into:
      gemm_epilogue (x, w, b?) -> (z)
        attrs: original gemm attrs + epilogue_bias/epilogue_act
    """
    out: List[Dict[str, Any]] = []
    i = 0

    def kind(op: Dict[str, Any]) -> str:
        return str(op.get("op", op.get("kind", ""))).strip().lower()

    while i < len(lowered):
        op0 = lowered[i]
        if kind(op0) != "gemm" or len(op0.get("outputs", [])) != 1:
            out.append(op0)
            i += 1
            continue

        gemm_out = int(op0["outputs"][0])

        j = i + 1
        bias_op = None
        if j < len(lowered) and kind(lowered[j]) == "bias_add":
            op1 = lowered[j]
            ins, outs = op1.get("inputs", []), op1.get("outputs", [])
            # must be epilogue-like: (gemm_out, bias) -> (gemm_out)
            if len(ins) == 2 and len(outs) == 1 and int(ins[0]) == gemm_out and int(outs[0]) == gemm_out:
                bias_op = op1
                j += 1

        relu_op = None
        if j < len(lowered) and kind(lowered[j]) == "relu":
            op2 = lowered[j]
            ins, outs = op2.get("inputs", []), op2.get("outputs", [])
            # relu consumes gemm_out (after optional in-place bias)
            if len(ins) == 1 and len(outs) == 1 and int(ins[0]) == gemm_out:
                relu_op = op2

        if relu_op is None:
            out.append(op0)
            i += 1
            continue

        fused_inputs = list(op0.get("inputs", []))  # [x, W]
        fused_attrs = dict(op0.get("attrs", {}) or {})

        fused_attrs["epilogue_act"] = "relu"
        if bias_op is not None:
            fused_inputs.append(int(bias_op["inputs"][1]))  # bias vid
            fused_attrs["epilogue_bias"] = True
        else:
            fused_attrs["epilogue_bias"] = False

        fused = {
            "op": "gemm_epilogue",
            "inputs": fused_inputs,
            "outputs": list(relu_op.get("outputs", [])),  # relu output becomes final
            "attrs": fused_attrs,
            # kernel_id는 지금 단계에선 비워도 됨 (fallback이 채우거나, 이후 KernelIndex가 채움)
            # "kernel_id": None,
        }

        out.append(fused)
        i = j + 1  # consume gemm + (bias_add?) + relu

    return out
