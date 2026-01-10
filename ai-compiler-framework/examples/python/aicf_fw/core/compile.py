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
# Lowering (your patched lowering 그대로)
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

    # -------------------------
    # Collect nodes by kind (preserve original order)
    # -------------------------
    linear_nodes = [n for n in ir.nodes if n.op == "Linear"]
    relu_nodes   = [n for n in ir.nodes if n.op == "ReLU"]
    mse_nodes    = [n for n in ir.nodes if n.op == "MseGrad"]
    step_nodes   = [n for n in ir.nodes if n.op == "StepInc"]
    bc_nodes     = [n for n in ir.nodes if n.op == "BiasCorr"]
    adam_nodes   = [n for n in ir.nodes if n.op == "AdamStep"]
    bwd_nodes    = [n for n in ir.nodes if n.op == "Backward"]

    # -------------------------
    # Helpers for grad vid allocation / picking
    # -------------------------
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
        cands = [int(vid) for vid, val in ir.values.items()
                 if getattr(val, "name", "") == "grad" and tuple(val.shape) == shape]
        for gid in cands:
            if gid not in used_grad:
                used_grad.add(gid)
                return gid
        gid = _alloc_grad_vid(shape, like_vid=like_vid)
        used_grad.add(gid)
        return gid

    # -------------------------
    # 1) Forward lowering (Linear/ReLU) in original IR order
    #    but we SKIP StepInc/BiasCorr/AdamStep/Backward for now.
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

        # delay these
        if op in ("Backward", "StepInc", "BiasCorr", "AdamStep"):
            continue

        lowered.append({"op": f"UNSUPPORTED<{op}>", "attrs": dict(n.attrs), "inputs": list(n.inputs), "outputs": list(n.outputs)})

    # -------------------------
    # 2) Backward lowering (v0: 2 Linear + 1 ReLU + mse_grad pattern)
    #    This MUST come before optimizer.
    # -------------------------
    grad_map_param_to_grad: Dict[int, int] = {}  # param_vid -> grad_vid

    if bwd_nodes:
        if not mse_nodes or len(linear_nodes) < 2 or len(relu_nodes) < 1:
            raise RuntimeError("lower_to_backend_ops: Backward lowering expects Linear->ReLU->Linear + MseGrad")

        lin0 = linear_nodes[0]
        relu0 = relu_nodes[0]
        lin1 = linear_nodes[1]
        mse = mse_nodes[-1]

        out_grad_vid = int(mse.outputs[0])  # dLoss/dPred

        # forward vids
        x_vid = int(lin0.inputs[0])
        W0_vid = int(lin0.inputs[1])
        b0_vid = int(lin0.inputs[2]) if bool(lin0.attrs.get("bias", False)) else None
        lin0_out_vid = int(lin0.outputs[0])

        relu_out_vid = int(relu0.outputs[0])

        W1_vid = int(lin1.inputs[1])
        b1_vid = int(lin1.inputs[2]) if bool(lin1.attrs.get("bias", False)) else None

        # param grads (allocate grads that backward writes into)
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

        # internal grads
        d_relu_out_vid = pick_next_grad(tuple(v(relu_out_vid).shape), like_vid=relu_out_vid)
        d_lin0_out_vid = pick_next_grad(tuple(v(lin0_out_vid).shape), like_vid=lin0_out_vid)
        dx0_vid = pick_next_grad(tuple(v(x_vid).shape), like_vid=x_vid)

        # ---- Linear1 backward: d_relu_out, dW1, db1 ----
        lowered.append({
            "op": "gemm",
            "attrs": {"transA": False, "transB": False},
            "inputs": [out_grad_vid, W1_vid],
            "outputs": [d_relu_out_vid],
        })
        lowered.append({
            "op": "gemm",
            "attrs": {"transA": True, "transB": False},
            "inputs": [out_grad_vid, relu_out_vid],
            "outputs": [dW1_vid],
        })
        if db1_vid is not None:
            lowered.append({
                "op": "reduce_sum",
                "attrs": {"axis": 0, "keepdim": False},
                "inputs": [out_grad_vid],
                "outputs": [db1_vid],
            })

        # ---- ReLU backward: need y_saved ----
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

        # ---- Linear0 backward: dx0, dW0, db0 ----
        lowered.append({
            "op": "gemm",
            "attrs": {"transA": False, "transB": False},
            "inputs": [d_lin0_out_vid, W0_vid],
            "outputs": [dx0_vid],
        })
        lowered.append({
            "op": "gemm",
            "attrs": {"transA": True, "transB": False},
            "inputs": [d_lin0_out_vid, x_vid],
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
    # 3) Optim lowering LAST (step_inc -> bias_corr -> adam_step*N)
    #    AdamStep grad input is wired from grad_map_param_to_grad.
    # -------------------------
    # StepInc
    for n in step_nodes:
        si = int(n.inputs[0]); so = int(n.outputs[0])
        lowered.append({"op": "step_inc", "attrs": {}, "inputs": [si], "outputs": [so]})

    # BiasCorr
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

    # AdamStep (use backward-produced grads)
    for n in adam_nodes:
        p_in = int(n.inputs[0])

        if p_in not in grad_map_param_to_grad:
            # fallback: if backward wasn't present, pick by shape
            g_in = pick_next_grad(tuple(v(p_in).shape), like_vid=p_in)
        else:
            g_in = grad_map_param_to_grad[p_in]

        m_in = int(n.inputs[2])
        v_in = int(n.inputs[3])
        bc1 = int(n.inputs[4])
        bc2 = int(n.inputs[5])

        p_out = int(n.outputs[0])
        m_out = int(n.outputs[1])
        v_out = int(n.outputs[2])

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
# NEW: build runtime env (vid -> torch.Tensor)
# -------------------------
def _build_runtime_env_from_artifact_context(ir: IRGraph, *, step_fn, lowered: List[Dict[str, Any]]) -> Dict[int, torch.Tensor]:
    """
    Build vid->torch.Tensor mapping by re-running step_fn once (NOT tracing),
    relying on the fact that model/optim/x/t live in closures and buffers are allocated by warmup.

    We cannot "discover" tensors purely from IR unless the runtime exposes binding API.
    So we intentionally:
      - expect step_fn's closure objects to own tensors (params, adam state, scalars, grads, x/t)
      - and we map by IRValue.name + shape + dtype + device heuristics,
        then patch SSA outputs by aliasing outputs to inputs for in-place ops.

    This is pragmatic and sufficient for PR5: IRExecutor executes lowered ops against real tensors.
    """
    # Run one real step to ensure all needed buffers exist + grads are materialized.
    step_fn()
    torch.cuda.synchronize()

    # Heuristic tensor pool:
    # We collect tensors from:
    #  - any Tensor wrappers seen via closure objects (model/optim/x/t)
    #  - torch scalar tensors (step_i32, bc1_inv, bc2_inv)
    # The PR5 test already attaches art.model/optim/x/t after compile, but inside compile_and_capture we don't have them.
    #
    # Therefore: we only build env AFTER caller attaches (model/optim/x/t) OR
    # you pass them through closure in step_fn (recommended).
    #
    # In your pipeline, module.compile has access to model/optim/x/t already; use that to call attach_env later if needed.
    #
    # Here we implement generic best-effort that works if those objects are reachable from step_fn.__closure__.

    def _collect_from_obj(obj, out: List[torch.Tensor]):
        # aicf Tensor wrapper
        if hasattr(obj, "data") and isinstance(getattr(obj, "data"), torch.Tensor):
            out.append(obj.data)
        # torch tensor
        if isinstance(obj, torch.Tensor):
            out.append(obj)
        # containers
        if isinstance(obj, (list, tuple)):
            for z in obj:
                _collect_from_obj(z, out)
        if isinstance(obj, dict):
            for z in obj.values():
                _collect_from_obj(z, out)
        # objects with __dict__
        if hasattr(obj, "__dict__"):
            for z in obj.__dict__.values():
                _collect_from_obj(z, out)

    pool: List[torch.Tensor] = []
    if step_fn.__closure__:
        for c in step_fn.__closure__:
            _collect_from_obj(c.cell_contents, pool)

    # dedup by data_ptr (tensor storage identity)
    uniq: Dict[int, torch.Tensor] = {}
    for t in pool:
        try:
            if t.is_cuda:
                uniq[int(t.data_ptr())] = t
        except Exception:
            pass
    tensors = list(uniq.values())

    def _match_tensor(val) -> Optional[torch.Tensor]:
        # match by (shape, dtype, device)
        shp = tuple(val.shape)
        dt = str(val.dtype)
        dev = str(val.device)
        for tt in tensors:
            if tuple(tt.shape) != shp:
                continue
            if str(tt.dtype) != dt:
                continue
            if str(tt.device) != dev:
                continue
            return tt
        return None

    env: Dict[int, torch.Tensor] = {}

    # First pass: bind everything we can by metadata
    for vid, val in ir.values.items():
        t = _match_tensor(val)
        if t is not None:
            env[int(vid)] = t

    # Second pass: ensure outputs exist (allocate missing buffers) based on lowering sequence
    # - For ops that produce grads/temps, if missing, allocate with same device/dtype/shape as IRValue
    for it in lowered:
        for ov in it.get("outputs", []):
            ov = int(ov)
            if ov in env:
                continue
            meta = ir.values.get(ov, None)
            if meta is None:
                continue
            # allocate capture-safe? (we are OUTSIDE capture; this is IRExecutor eager path only)
            # Use empty() for speed; grads should be zero-init where needed by kernels anyway.
            dtype = getattr(torch, str(meta.dtype).split(".")[-1], None)
            # fallback: parse like "torch.float32"
            if dtype is None:
                if "float32" in str(meta.dtype):
                    dtype = torch.float32
                elif "float16" in str(meta.dtype):
                    dtype = torch.float16
                elif "bfloat16" in str(meta.dtype):
                    dtype = torch.bfloat16
                elif "int32" in str(meta.dtype):
                    dtype = torch.int32
                else:
                    dtype = torch.float32
            device = torch.device(str(meta.device))
            env[ov] = torch.empty(tuple(meta.shape), device=device, dtype=dtype)

    # Third pass: SSA aliasing rules for in-place ops:
    # - bias_add: output y aliases input y
    # - grad_zero: output g aliases input g
    # - step_inc: output step aliases input step (in-place)
    # - bias_corr: outputs are distinct scalars (no alias)
    # - adam_step: outputs alias p/m/v inputs (in-place update)
    for it in lowered:
        op = it["op"]
        ins = list(it.get("inputs", []))
        outs = list(it.get("outputs", []))
        if op in ("bias_add", "grad_zero", "step_inc", "copy"):
            if ins and outs:
                env[int(outs[0])] = env[int(ins[0])]
        if op == "adam_step":
            # outputs: [p_out, m_out, v_out] alias inputs [p_in, g, m_in, v_in, ...]
            if len(ins) >= 4 and len(outs) >= 3:
                env[int(outs[0])] = env[int(ins[0])]
                env[int(outs[1])] = env[int(ins[2])]
                env[int(outs[2])] = env[int(ins[3])]

    return env


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
) -> CompileArtifact:
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

    # NEW: attach env for IRExecutor
    try:
        env = _build_runtime_env_from_artifact_context(ir, step_fn=step_fn, lowered=lowered)
        art.attach_env(env)
    except Exception as e:
        # allow capture artifact to exist even if env build failed
        # but IRExecutor will fail with a clear error if env is empty
        print(f"[WARN] compile_and_capture: failed to build IRExecutor env: {e}")

    return art


__all__ = ["compile_ir", "lower_to_backend_ops", "compile_and_capture"]
