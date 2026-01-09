# aicf_fw/core/compile.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .ir import IRGraph
from .trace import tracing
from .warmup import warmup_capture_safe
from .validate import validate_ir
from .artifact import CompileArtifact
from aicf_fw.backend import get_backend


def compile_ir(step_fn, *, name: str = "train_step") -> IRGraph:
    """
    Run step_fn once in tracing mode and return IRGraph.
    step_fn must close over model/optim/x/t etc.
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
      Linear    -> gemm(transB=True) + (bias_add if bias)
      ReLU      -> relu
      MseGrad   -> mse_grad
      StepInc   -> step_inc
      BiasCorr  -> bias_corr
      AdamStep  -> adam_step
      Backward  -> (no direct runtime op here; gradients are lowered via traced ops like relu_bwd, etc)
    """
    lowered: List[Dict[str, Any]] = []

    for n in ir.nodes:
        op = n.op

        if op == "Linear":
            lowered.append({"op": "gemm", "attrs": {"transB": True}})
            if bool(n.attrs.get("bias", False)):
                lowered.append({"op": "bias_add", "attrs": {}})
            continue

        if op == "ReLU":
            lowered.append({"op": "relu", "attrs": {}})
            continue

        if op == "MseGrad":
            attrs: Dict[str, Any] = {}
            if "scale" in n.attrs:
                attrs["scale"] = float(n.attrs["scale"])
            lowered.append({"op": "mse_grad", "attrs": attrs})
            continue

        if op == "StepInc":
            lowered.append({"op": "step_inc", "attrs": {}})
            continue

        if op == "BiasCorr":
            lowered.append(
                {
                    "op": "bias_corr",
                    "attrs": {
                        "beta1": float(n.attrs["beta1"]),
                        "beta2": float(n.attrs["beta2"]),
                    },
                }
            )
            continue

        if op == "AdamStep":
            lowered.append(
                {
                    "op": "adam_step",
                    "attrs": {
                        "lr": float(n.attrs["lr"]),
                        "beta1": float(n.attrs["beta1"]),
                        "beta2": float(n.attrs["beta2"]),
                        "eps": float(n.attrs["eps"]),
                    },
                }
            )
            continue

        if op == "Backward":
            # Backward is a graph marker in v0; runtime ops are emitted by autograd + backend dispatch.
            continue

        # If you add more ops later, extend here.
        lowered.append({"op": f"UNSUPPORTED<{op}>", "attrs": dict(n.attrs)})

    return lowered


def compile_and_capture(
    step_fn,
    *,
    name: str = "train_step",
    warmup_runs: int = 2,
    warmup_sync: bool = True,
    validate: bool = True,
    trace: bool = True,
    enforce_ops: Sequence[str] = ("adam_step",),
    torch_sync: bool = True,
) -> CompileArtifact:
    """
    High-level entry:
      1) compile_ir
      2) (optional) validate_ir
      3) lower_to_backend_ops
      4) warmup_capture_safe to materialize buffers (capture-safe)
      5) backend capture + runtime trace
      6) return CompileArtifact(name, ir, lowered, trace_ops, backend)

    Assumptions:
      - backend implements capture_reset/begin/end, trace_reset/enable/get, replay()
      - step_fn is capture-safe after warmup (no dynamic alloc/branch)
    """
    # 1) IR compile
    ir = compile_ir(step_fn, name=name)

    # 2) validate IR invariants
    if validate:
        report = validate_ir(ir)
        # Keep warnings visible but non-fatal (same spirit as PR3)
        for w in report.warnings:
            print(f"[WARN] {w}")

    # 3) Lowering plan
    lowered = lower_to_backend_ops(ir)

    # 4) Warmup
    if warmup_runs and warmup_runs > 0:
        warmup_capture_safe(train_step=step_fn, runs=warmup_runs, sync=warmup_sync)

    # 5) Capture + trace
    backend = get_backend()

    backend.capture_reset()
    if torch_sync:
        torch.cuda.synchronize()

    backend.trace_reset()
    backend.trace_enable(bool(trace))

    backend.capture_begin()
    step_fn()
    backend.capture_end()

    if torch_sync:
        torch.cuda.synchronize()

    trace_ops: List[str] = backend.trace_get() if trace else []
    backend.trace_enable(False)

    art = CompileArtifact(name=name, ir=ir, lowered=lowered, trace_ops=trace_ops, backend=backend)

    # 6) enforce required runtime ops (PR3: adam_step must exist)
    for op in enforce_ops:
        art.assert_trace_has(op)

    return art
