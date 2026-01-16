from __future__ import annotations
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

def compile_train_step(model, optimizer, *, B: int, D: int, device: str | torch.device, dtype: torch.dtype, name: str = "fw_train_step"):
    dev = torch.device(device) if isinstance(device, str) else device

    named_params = list(model.named_parameters())
    assert len(named_params) > 0

    # ----- runtime bindings -----
    params_rt: dict[str, torch.Tensor] = {pname: p for pname, p in named_params}
    statics_rt: dict[str, torch.Tensor] = optimizer.named_state_tensors()

    # ----- trace build fn -----
    def build():
        # inputs
        sx = sym_tensor(name="x", shape=(B, D), dtype=dtype, device=dev)
        st = sym_tensor(name="t", shape=(B, D), dtype=dtype, device=dev)

        # param syms
        psym = {pname: sym_tensor(name=pname, shape=tuple(p.shape), dtype=dtype, device=dev) for pname, p in named_params}

        # optimizer syms
        s_bc1 = sym_tensor(name=BC1_NAME, shape=(), dtype=dtype, device=dev)
        s_bc2 = sym_tensor(name=BC2_NAME, shape=(), dtype=dtype, device=dev)

        msym = {pname: sym_tensor(name=opt_m_name(pname), shape=tuple(p.shape), dtype=dtype, device=dev) for pname, p in named_params}
        vsym = {pname: sym_tensor(name=opt_v_name(pname), shape=tuple(p.shape), dtype=dtype, device=dev) for pname, p in named_params}

        # --- forward ---
        # We also need ReLU saved tensor for relu_bwd.
        # Convention: the ReLU module emits save(name="<prefix>.relu_saved") but we don't get a handle back.
        # MVP: re-create the save explicitly here at compile-time, by pattern:
        #
        # For Sequential(Linear, ReLU, Linear): we do:
        #   lin0 -> relu0 -> save(relu0) -> lin1 -> mse_grad
        #
        # If you later generalize, you'll want a real autograd tape.

        # assume: model is exactly Linear(0), ReLU(1), Linear(2)
        # get param names from registry
        # 0.W,0.b,2.W,2.b style
        W0 = psym["0.W"]; b0 = psym["0.b"]
        W1 = psym["2.W"]; b1 = psym["2.b"]

        lin0 = model._modules["0"].forward_ir(sx, psym)  # uses 0.W/0.b
        relu0 = model._modules["1"].forward_ir(lin0, psym)
        relu0_saved = save(relu0, name="relu0_saved")    # explicit handle
        lin1 = model._modules["2"].forward_ir(relu0, psym)

        dY = mse_grad(lin1, st, name="dY")

        # --- backward ---
        # linear1 bwd
        d_relu0, dW1, db1 = linear_bwd(
            relu0, W1, dY,
            bias=True,
            dx_name="d_relu0_out",
            dW_name="d_2.W",
            db_name="d_2.b",
        )
        # relu bwd
        d_lin0 = relu_bwd(d_relu0, relu0_saved, name="d_lin0_out")
        # linear0 bwd
        _dx, dW0, db0 = linear_bwd(
            sx, W0, d_lin0,
            bias=True,
            dx_name="d_x",
            dW_name="d_0.W",
            db_name="d_0.b",
        )

        # --- adam in-place ---
        adam_step(W0, dW0, msym["0.W"], vsym["0.W"], s_bc1, s_bc2, lr=optimizer.lr, beta1=optimizer.beta1, beta2=optimizer.beta2, eps=optimizer.eps)
        adam_step(b0, db0, msym["0.b"], vsym["0.b"], s_bc1, s_bc2, lr=optimizer.lr, beta1=optimizer.beta1, beta2=optimizer.beta2, eps=optimizer.eps)
        adam_step(W1, dW1, msym["2.W"], vsym["2.W"], s_bc1, s_bc2, lr=optimizer.lr, beta1=optimizer.beta1, beta2=optimizer.beta2, eps=optimizer.eps)
        adam_step(b1, db1, msym["2.b"], vsym["2.b"], s_bc1, s_bc2, lr=optimizer.lr, beta1=optimizer.beta1, beta2=optimizer.beta2, eps=optimizer.eps)

    ir = trace_ir(build, name=name)
    lowered = lower_to_backend_ops(ir)
    plan = build_binding_plan(ir)
    ex = PlannedExecutor(ir=ir, lowered=lowered, plan=plan, opts=ExecOptions(debug=False))

    return CompiledTrainStep(ir=ir, lowered=lowered, plan=plan, ex=ex, params=params_rt, statics=statics_rt, optimizer=optimizer)
