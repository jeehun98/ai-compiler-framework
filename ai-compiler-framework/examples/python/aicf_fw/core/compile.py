# aicf_fw/core/compile.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import torch

from .ir import IRGraph
from .trace import tracing
from aicf_fw.backend import get_backend


def compile_ir(step_fn, *, name: str = "train_step") -> IRGraph:
    """
    Run step_fn once in tracing mode and return IRGraph.
    step_fn must close over model/optim/x/t etc (like your PR1 test).
    """
    ir = IRGraph(name=name)
    with tracing(ir):
        step_fn()
    return ir


def lower_to_backend_ops(ir: IRGraph) -> List[Dict[str, Any]]:
    """
    Lower high-level IR ops to backend op_name sequence.
    Returns list of {op, attrs} in execution order.
    v0 lowering rules:
      Linear -> gemm(transB=True) + (bias_add if bias)
      ReLU   -> relu
      MseGrad-> mse_grad
      StepInc-> step_inc
      BiasCorr-> bias_corr
      AdamStep-> adam_step
    """
    lowered: List[Dict[str, Any]] = []

    for n in ir.nodes:
        op = n.op
        if op == "Linear":
            # inputs: x, W, (b?)
            # runtime path is: gemm(transB=True), then bias_add in-place if bias
            lowered.append({"op": "gemm", "attrs": {"transB": True}})
            if bool(n.attrs.get("bias", False)):
                lowered.append({"op": "bias_add", "attrs": {}})
            continue

        if op == "ReLU":
            lowered.append({"op": "relu", "attrs": {}})
            continue

        if op == "MseGrad":
            attrs = {}
            if "scale" in n.attrs:
                attrs["scale"] = float(n.attrs["scale"])
            lowered.append({"op": "mse_grad", "attrs": attrs})
            continue

        if op == "StepInc":
            lowered.append({"op": "step_inc", "attrs": {}})
            continue

        if op == "BiasCorr":
            lowered.append({"op": "bias_corr", "attrs": {"beta1": float(n.attrs["beta1"]), "beta2": float(n.attrs["beta2"])}})
            continue

        if op == "AdamStep":
            lowered.append({"op": "adam_step", "attrs": {"lr": float(n.attrs["lr"]), "beta1": float(n.attrs["beta1"]), "beta2": float(n.attrs["beta2"]), "eps": float(n.attrs["eps"])}})
            continue
        
        if op == "Backward":
            continue


        # If you add more ops later, extend here.
        lowered.append({"op": f"UNSUPPORTED<{op}>", "attrs": dict(n.attrs)})

    return lowered
