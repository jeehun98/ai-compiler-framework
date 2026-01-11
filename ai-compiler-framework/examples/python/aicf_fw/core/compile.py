# aicf_fw/core/compile.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

import torch

from .ir import IRGraph
from .trace import tracing
from .warmup import warmup_capture_safe
from .validate import validate_ir
from .artifact import CompileArtifact
from aicf_fw.backend import get_backend


# -------------------------
# IR compile
# -------------------------
def compile_ir(step_fn, *, name: str = "train_step") -> IRGraph:
    ir = IRGraph(name=name)
    with tracing(ir):
        step_fn()
    return ir


# -------------------------
# Lowering (patched lowering)
# -------------------------
def lower_to_backend_ops(ir) -> List[Dict[str, Any]]:
    """
    Fixed lowering order:
      forward -> mse_grad -> backward ops -> (step_inc -> bias_corr -> adam_step * N)

    Also fixes AdamStep grad wiring:
      adam_step grad input MUST be the grad value produced by backward lowering.
    """
    lowered: List[Dict[str, Any]] = []

    def v(vid: int):
        return ir.values[int(vid)]

    linear_nodes = [n for n in ir.nodes if n.op == "Linear"]
    relu_nodes   = [n for n in ir.nodes if n.op == "ReLU"]
    mse_nodes    = [n for n in ir.nodes if n.op == "MseGrad"]
    step_nodes   = [n for n in ir.nodes if n.op == "StepInc"]
    bc_nodes     = [n for n in ir.nodes if n.op == "BiasCorr"]
    adam_nodes   = [n for n in ir.nodes if n.op == "AdamStep"]
    bwd_nodes    = [n for n in ir.nodes if n.op == "Backward"]

    used_grad: Set[int] = set()

    def _alloc_grad_vid(shape: Tuple[int, ...], *, like_vid: int) -> int:
        ref = v(like_vid)
        g = ir.new_value(
            name="grad",
            shape=tuple(shape),
            dtype=str(ref.dtype),
            device=str(ref.device),
        )
        return int(g.id)

    def pick_next_grad(shape: Tuple[int, ...], *, like_vid: int) -> int:
        shape = tuple(shape)
        cands = [
            int(vid) for vid, val in ir.values.items()
            if getattr(val, "name", "") == "grad" and tuple(val.shape) == shape
        ]
        for gid in cands:
            if gid not in used_grad:
                used_grad.add(gid)
                return gid
        gid = _alloc_grad_vid(shape, like_vid=like_vid)
        used_grad.add(gid)
        return gid

    # -------------------------
    # 1) Forward lowering
    # -------------------------
    for n in ir.nodes:
        op = n.op

        if op == "Linear":
            x_vid = int(n.inputs[0])
            W_vid = int(n.inputs[1])
            y_vid = int(n.outputs[0])
            lowered.append({"op": "gemm", "attrs": {"transB": True}, "inputs": [x_vid, W_vid], "outputs": [y_vid]})
            if bool(n.attrs.get("bias", False)):
                b_vid = int(n.inputs[2])
                lowered.append({"op": "bias_add", "attrs": {}, "inputs": [y_vid, b_vid], "outputs": [y_vid]})
            continue

        if op == "ReLU":
            x_vid = int(n.inputs[0])
            y_vid = int(n.outputs[0])
            lowered.append({"op": "relu", "attrs": {}, "inputs": [x_vid], "outputs": [y_vid]})
            continue

        if op == "MseGrad":
            pred_vid = int(n.inputs[0])
            tgt_vid  = int(n.inputs[1])
            out_vid  = int(n.outputs[0])
            attrs: Dict[str, Any] = {}
            if "scale" in n.attrs:
                attrs["scale"] = float(n.attrs["scale"])
            lowered.append({"op": "mse_grad", "attrs": attrs, "inputs": [pred_vid, tgt_vid], "outputs": [out_vid]})
            continue

        if op in ("Backward", "StepInc", "BiasCorr", "AdamStep"):
            continue

        lowered.append({
            "op": f"UNSUPPORTED<{op}>",
            "attrs": dict(n.attrs),
            "inputs": list(n.inputs),
            "outputs": list(n.outputs),
        })

    # -------------------------
    # 2) Backward lowering (v0 pattern)
    # -------------------------
    grad_map_param_to_grad: Dict[int, int] = {}  # param_vid -> grad_vid

    if bwd_nodes:
        if not mse_nodes or len(linear_nodes) < 2 or len(relu_nodes) < 1:
            raise RuntimeError("lower_to_backend_ops: Backward expects Linear->ReLU->Linear + MseGrad")

        lin0 = linear_nodes[0]
        relu0 = relu_nodes[0]
        lin1 = linear_nodes[1]
        mse = mse_nodes[-1]

        out_grad_vid = int(mse.outputs[0])  # dLoss/dPred

        x_vid = int(lin0.inputs[0])
        W0_vid = int(lin0.inputs[1])
        b0_vid = int(lin0.inputs[2]) if bool(lin0.attrs.get("bias", False)) else None
        lin0_out_vid = int(lin0.outputs[0])

        relu_out_vid = int(relu0.outputs[0])

        W1_vid = int(lin1.inputs[1])
        b1_vid = int(lin1.inputs[2]) if bool(lin1.attrs.get("bias", False)) else None

        dW0_vid = pick_next_grad(tuple(v(W0_vid).shape), like_vid=W0_vid)
        grad_map_param_to_grad[W0_vid] = dW0_vid

        if b0_vid is not None:
            db0_vid = pick_next_grad(tuple(v(b0_vid).shape), like_vid=b0_vid)
            grad_map_param_to_grad[b0_vid] = db0_vid
        else:
            db0_vid = None

        dW1_vid = pick_next_grad(tuple(v(W1_vid).shape), like_vid=W1_vid)
        grad_map_param_to_grad[W1_vid] = dW1_vid

        if b1_vid is not None:
            db1_vid = pick_next_grad(tuple(v(b1_vid).shape), like_vid=b1_vid)
            grad_map_param_to_grad[b1_vid] = db1_vid
        else:
            db1_vid = None

        d_relu_out_vid = pick_next_grad(tuple(v(relu_out_vid).shape), like_vid=relu_out_vid)
        d_lin0_out_vid = pick_next_grad(tuple(v(lin0_out_vid).shape), like_vid=lin0_out_vid)
        dx0_vid = pick_next_grad(tuple(v(x_vid).shape), like_vid=x_vid)

        # ---- Linear1 backward ----
        # d_relu_out = out_grad @ W1
        lowered.append({
            "op": "gemm",
            "attrs": {"transA": False, "transB": False},
            "inputs": [out_grad_vid, W1_vid],
            "outputs": [d_relu_out_vid],
        })
        # dW1 = relu_out^T @ out_grad   (SWAP FIX)
        lowered.append({
            "op": "gemm",
            "attrs": {"transA": True, "transB": False},
            "inputs": [relu_out_vid, out_grad_vid],  # <-- swapped
            "outputs": [dW1_vid],
        })
        if db1_vid is not None:
            lowered.append({
                "op": "reduce_sum",
                "attrs": {"axis": 0, "keepdim": False},
                "inputs": [out_grad_vid],
                "outputs": [db1_vid],
            })

        # ---- ReLU backward ----
        relu_y_saved_vid = None
        for vid, val in ir.values.items():
            if getattr(val, "name", "") == "relu_y_saved" and tuple(val.shape) == tuple(v(relu_out_vid).shape):
                relu_y_saved_vid = int(vid)
                break
        if relu_y_saved_vid is None:
            saved = ir.new_value(
                name="relu_y_saved",
                shape=tuple(v(relu_out_vid).shape),
                dtype=str(v(relu_out_vid).dtype),
                device=str(v(relu_out_vid).device),
            )
            relu_y_saved_vid = int(saved.id)

        lowered.append({"op": "copy", "attrs": {}, "inputs": [relu_out_vid], "outputs": [relu_y_saved_vid]})
        lowered.append({
            "op": "relu_bwd",
            "attrs": {},
            "inputs": [d_relu_out_vid, relu_y_saved_vid],
            "outputs": [d_lin0_out_vid],
        })

        # ---- Linear0 backward ----
        # dx0 = d_lin0_out @ W0
        lowered.append({
            "op": "gemm",
            "attrs": {"transA": False, "transB": False},
            "inputs": [d_lin0_out_vid, W0_vid],
            "outputs": [dx0_vid],
        })
        # dW0 = x^T @ d_lin0_out   (SWAP FIX)
        lowered.append({
            "op": "gemm",
            "attrs": {"transA": True, "transB": False},
            "inputs": [x_vid, d_lin0_out_vid],  # <-- swapped
            "outputs": [dW0_vid],
        })
        if db0_vid is not None:
            lowered.append({
                "op": "reduce_sum",
                "attrs": {"axis": 0, "keepdim": False},
                "inputs": [d_lin0_out_vid],
                "outputs": [db0_vid],
            })

    # -------------------------
    # 3) Optim lowering LAST
    # -------------------------
    for n in step_nodes:
        si = int(n.inputs[0])
        so = int(n.outputs[0])
        lowered.append({"op": "step_inc", "attrs": {}, "inputs": [si], "outputs": [so]})

    for n in bc_nodes:
        step_vid = int(n.inputs[0])
        b1_vid = int(n.outputs[0])
        b2_vid = int(n.outputs[1])
        lowered.append({
            "op": "bias_corr",
            "attrs": {"beta1": float(n.attrs["beta1"]), "beta2": float(n.attrs["beta2"])},
            "inputs": [step_vid],
            "outputs": [b1_vid, b2_vid],
        })

    for n in adam_nodes:
        p_in = int(n.inputs[0])

        if p_in not in grad_map_param_to_grad:
            g_in = pick_next_grad(tuple(v(p_in).shape), like_vid=p_in)
        else:
            g_in = grad_map_param_to_grad[p_in]

        m_in = int(n.inputs[2])
        v_in = int(n.inputs[3])
        bc1 = int(n.inputs[4])
        bc2 = int(n.inputs[5])

        # ★ outputs 강제 in-place
        p_out = p_in
        m_out = m_in
        v_out = v_in

        lowered.append({
            "op": "adam_step",
            "attrs": {
                "lr": float(n.attrs["lr"]),
                "beta1": float(n.attrs["beta1"]),
                "beta2": float(n.attrs["beta2"]),
                "eps": float(n.attrs["eps"]),
            },
            "inputs": [p_in, g_in, m_in, v_in, bc1, bc2],
            "outputs": [p_out, m_out, v_out],
        })


    return lowered


# -------------------------
# compile + capture
# -------------------------
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
    # NOTE: accepted for compatibility with Module.compile() but unused here.
    bind_model: Optional[Any] = None,
    bind_optim: Optional[Any] = None,
    bind_x: Optional[Any] = None,
    bind_t: Optional[Any] = None,
) -> CompileArtifact:
    """
    compile -> validate -> lower -> warmup -> capture -> trace

    IMPORTANT:
      - This function DOES NOT build/attach IRExecutor env anymore.
      - Env binding is done by Module.compile() (exact binding) or by user code.
    """
    ir = compile_ir(step_fn, name=name)

    if validate:
        report = validate_ir(ir)
        for w in report.warnings:
            print(f"[WARN] {w}")

    lowered = lower_to_backend_ops(ir)

    if warmup_runs and warmup_runs > 0:
        warmup_capture_safe(train_step=step_fn, runs=warmup_runs, sync=warmup_sync)

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

    for op in enforce_ops:
        art.assert_trace_has(op)

    return art


__all__ = ["compile_ir", "lower_to_backend_ops", "compile_and_capture"]
