from __future__ import annotations
import os
import torch

from aicf_fw.core_v2 import trace_ir
from aicf_fw.core_v2.lower import lower_to_backend_ops
from aicf_fw.core_v2.plan import build_binding_plan
from aicf_fw.core_v2.exec import PlannedExecutor, ExecOptions
from aicf_fw.core_v2.ops import (
    sym_tensor,
    linear_bwd,
    relu_bwd,
    mse_grad,
    save,
    adam_step,
)

from aicf_fw.fw.naming import BC1_NAME, BC2_NAME, opt_m_name, opt_v_name
from aicf_fw.fw.train_step import CompiledTrainStep

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    if v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

def compile_train_step(
    model,
    optimizer,
    *,
    B: int,
    D: int,
    device: str | torch.device,
    dtype: torch.dtype,
    name: str = "fw_train_step",
    warmup_runs: int | None = None,
    warmup_inputs: dict[str, torch.Tensor] | None = None,
    warmup_required: bool = True,
):
    dev = torch.device(device) if isinstance(device, str) else device

    # warmup policy:
    # - default warmup_runs=2
    # - allow env override AICF_WARMUP (int)
    if warmup_runs is None:
        warmup_runs = _env_int("AICF_WARMUP", 2)
    warmup_runs = int(warmup_runs)

    named_params = list(model.named_parameters())
    assert len(named_params) > 0

    # runtime bindings
    params_rt: dict[str, torch.Tensor] = {pname: p for pname, p in named_params}
    statics_rt: dict[str, torch.Tensor] = optimizer.named_state_tensors()

    # trace build fn (MVP: Sequential(Linear, ReLU, Linear) + MSE + Adam)
    def build():
        sx = sym_tensor(name="x", shape=(B, D), dtype=dtype, device=dev)
        st = sym_tensor(name="t", shape=(B, D), dtype=dtype, device=dev)

        psym = {pname: sym_tensor(name=pname, shape=tuple(p.shape), dtype=dtype, device=dev) for pname, p in named_params}

        s_bc1 = sym_tensor(name=BC1_NAME, shape=(), dtype=dtype, device=dev)
        s_bc2 = sym_tensor(name=BC2_NAME, shape=(), dtype=dtype, device=dev)

        msym = {pname: sym_tensor(name=opt_m_name(pname), shape=tuple(p.shape), dtype=dtype, device=dev) for pname, p in named_params}
        vsym = {pname: sym_tensor(name=opt_v_name(pname), shape=tuple(p.shape), dtype=dtype, device=dev) for pname, p in named_params}

        # MVP hard-assumption: model is Sequential with modules "0"(Linear), "1"(ReLU), "2"(Linear)
        lin0 = model._modules["0"].forward_ir(sx, psym)
        relu0 = model._modules["1"].forward_ir(lin0, psym)
        relu0_saved = save(relu0, name="relu0_saved")
        lin1 = model._modules["2"].forward_ir(relu0, psym)

        dY = mse_grad(lin1, st, name="dY")

        # grab param syms by expected names
        W0 = psym["0.W"]; b0 = psym["0.b"]
        W1 = psym["2.W"]; b1 = psym["2.b"]

        d_relu0, dW1, db1 = linear_bwd(
            relu0, W1, dY,
            bias=True,
            dx_name="d_relu0_out",
            dW_name="d_2.W",
            db_name="d_2.b",
        )
        d_lin0 = relu_bwd(d_relu0, relu0_saved, name="d_lin0_out")
        _dx, dW0, db0 = linear_bwd(
            sx, W0, d_lin0,
            bias=True,
            dx_name="d_x",
            dW_name="d_0.W",
            db_name="d_0.b",
        )

        # in-place adam updates
        adam_step(W0, dW0, msym["0.W"], vsym["0.W"], s_bc1, s_bc2, lr=optimizer.lr, beta1=optimizer.beta1, beta2=optimizer.beta2, eps=optimizer.eps)
        adam_step(b0, db0, msym["0.b"], vsym["0.b"], s_bc1, s_bc2, lr=optimizer.lr, beta1=optimizer.beta1, beta2=optimizer.beta2, eps=optimizer.eps)
        adam_step(W1, dW1, msym["2.W"], vsym["2.W"], s_bc1, s_bc2, lr=optimizer.lr, beta1=optimizer.beta1, beta2=optimizer.beta2, eps=optimizer.eps)
        adam_step(b1, db1, msym["2.b"], vsym["2.b"], s_bc1, s_bc2, lr=optimizer.lr, beta1=optimizer.beta1, beta2=optimizer.beta2, eps=optimizer.eps)

    ir = trace_ir(build, name=name)
    lowered = lower_to_backend_ops(ir)
    plan = build_binding_plan(ir)
    ex = PlannedExecutor(ir=ir, lowered=lowered, plan=plan, opts=ExecOptions(debug=False))

    compiled = CompiledTrainStep(
        ir=ir,
        lowered=lowered,
        plan=plan,
        ex=ex,
        params=params_rt,
        statics=statics_rt,
        optimizer=optimizer,
        warmup_runs=warmup_runs,
        warmup_required=warmup_required,
    )

    # optional: auto-warmup at compile time if inputs provided
    if warmup_inputs is not None and warmup_runs > 0:
        compiled.warmup(warmup_inputs, n=warmup_runs, reuse_static=True)

    return compiled
