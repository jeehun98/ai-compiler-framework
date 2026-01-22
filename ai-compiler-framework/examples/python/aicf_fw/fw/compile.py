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


def _infer_from_warmup_inputs(
    warmup_inputs: dict[str, torch.Tensor] | None,
    *,
    B: int | None,
    D: int | None,
    device: str | torch.device | None,
    dtype: torch.dtype | None,
    model,
) -> tuple[int, int, torch.device, torch.dtype]:
    """
    Infer (B, D, device, dtype) from warmup_inputs["x"] and validate against:
      - warmup_inputs["t"]
      - model parameters
      - user-provided B/D/device/dtype
    """
    if warmup_inputs is None:
        raise ValueError("compile_train_step: warmup_inputs is required when B/D/device/dtype are not fully provided.")

    if "x" not in warmup_inputs or "t" not in warmup_inputs:
        raise ValueError("compile_train_step: warmup_inputs must contain both 'x' and 't'.")

    x = warmup_inputs["x"]
    t = warmup_inputs["t"]

    # ---- x / t validation ----
    for name, v in (("x", x), ("t", t)):
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"compile_train_step: warmup_inputs['{name}'] must be a torch.Tensor.")
        if v.ndim != 2:
            raise ValueError(
                f"compile_train_step: warmup_inputs['{name}'] must be rank-2 (B,D). got shape={tuple(v.shape)}"
            )

    if tuple(x.shape) != tuple(t.shape):
        raise ValueError(
            f"compile_train_step: x and t shape mismatch. x={tuple(x.shape)} vs t={tuple(t.shape)}"
        )

    B_inf, D_inf = int(x.shape[0]), int(x.shape[1])
    dev_inf = x.device
    dt_inf = x.dtype

    # ---- user arg validation ----
    if B is not None and int(B) != B_inf:
        raise ValueError(f"compile_train_step: B mismatch. arg B={B} vs warmup x.shape[0]={B_inf}")
    if D is not None and int(D) != D_inf:
        raise ValueError(f"compile_train_step: D mismatch. arg D={D} vs warmup x.shape[1]={D_inf}")

    if device is not None:
        dev_arg = torch.device(device) if isinstance(device, str) else device
        if dev_arg != dev_inf:
            raise ValueError(
                f"compile_train_step: device mismatch. arg device={dev_arg} vs warmup x.device={dev_inf}"
            )

    if dtype is not None and dtype != dt_inf:
        raise ValueError(
            f"compile_train_step: dtype mismatch. arg dtype={dtype} vs warmup x.dtype={dt_inf}"
        )

    # ---- model parameter cross-check ----
    for pname, p in model.named_parameters():
        if p.device != dev_inf:
            raise ValueError(
                f"compile_train_step: parameter '{pname}' device mismatch. "
                f"param.device={p.device} vs input.device={dev_inf}"
            )
        if p.dtype != dt_inf:
            raise ValueError(
                f"compile_train_step: parameter '{pname}' dtype mismatch. "
                f"param.dtype={p.dtype} vs input.dtype={dt_inf}"
            )

    B_out = B_inf if B is None else int(B)
    D_out = D_inf if D is None else int(D)
    dev_out = dev_inf if device is None else (torch.device(device) if isinstance(device, str) else device)
    dt_out = dt_inf if dtype is None else dtype

    return B_out, D_out, dev_out, dt_out


def compile_train_step(
    model,
    optimizer,
    *,
    B: int | None = None,
    D: int | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    name: str = "fw_train_step",
    warmup_runs: int | None = None,
    warmup_inputs: dict[str, torch.Tensor] | None = None,
    warmup_required: bool = True,
):
    # ---- warmup policy ----
    if warmup_runs is None:
        warmup_runs = _env_int("AICF_WARMUP", 2)
    warmup_runs = int(warmup_runs)

    # ---- infer / validate runtime shape & type ----
    if (B is None) or (D is None) or (device is None) or (dtype is None):
        B, D, dev, dtype = _infer_from_warmup_inputs(
            warmup_inputs,
            B=B,
            D=D,
            device=device,
            dtype=dtype,
            model=model,
        )
    else:
        dev = torch.device(device) if isinstance(device, str) else device

    named_params = list(model.named_parameters())
    assert len(named_params) > 0

    # ---- runtime bindings ----
    params_rt: dict[str, torch.Tensor] = {pname: p for pname, p in named_params}
    statics_rt: dict[str, torch.Tensor] = optimizer.named_state_tensors()

    # ---- trace build fn ----
    def build():
        sx = sym_tensor(name="x", shape=(B, D), dtype=dtype, device=dev)
        st = sym_tensor(name="t", shape=(B, D), dtype=dtype, device=dev)

        psym = {
            pname: sym_tensor(
                name=pname,
                shape=tuple(p.shape),
                dtype=dtype,
                device=dev,
            )
            for pname, p in named_params
        }

        s_bc1 = sym_tensor(name=BC1_NAME, shape=(), dtype=dtype, device=dev)
        s_bc2 = sym_tensor(name=BC2_NAME, shape=(), dtype=dtype, device=dev)

        msym = {
            pname: sym_tensor(
                name=opt_m_name(pname),
                shape=tuple(p.shape),
                dtype=dtype,
                device=dev,
            )
            for pname, p in named_params
        }
        vsym = {
            pname: sym_tensor(
                name=opt_v_name(pname),
                shape=tuple(p.shape),
                dtype=dtype,
                device=dev,
            )
            for pname, p in named_params
        }

        # MVP assumption: Sequential(Linear, ReLU, Linear)
        lin0 = model._modules["0"].forward_ir(sx, psym)
        relu0 = model._modules["1"].forward_ir(lin0, psym)
        relu0_saved = save(relu0, name="relu0_saved")
        lin1 = model._modules["2"].forward_ir(relu0, psym)

        dY = mse_grad(lin1, st, name="dY")

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

        adam_step(W0, dW0, msym["0.W"], vsym["0.W"], s_bc1, s_bc2,
                  lr=optimizer.lr, beta1=optimizer.beta1, beta2=optimizer.beta2, eps=optimizer.eps)
        adam_step(b0, db0, msym["0.b"], vsym["0.b"], s_bc1, s_bc2,
                  lr=optimizer.lr, beta1=optimizer.beta1, beta2=optimizer.beta2, eps=optimizer.eps)
        adam_step(W1, dW1, msym["2.W"], vsym["2.W"], s_bc1, s_bc2,
                  lr=optimizer.lr, beta1=optimizer.beta1, beta2=optimizer.beta2, eps=optimizer.eps)
        adam_step(b1, db1, msym["2.b"], vsym["2.b"], s_bc1, s_bc2,
                  lr=optimizer.lr, beta1=optimizer.beta1, beta2=optimizer.beta2, eps=optimizer.eps)

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

    if warmup_inputs is not None and warmup_runs > 0:
        compiled.warmup(warmup_inputs, n=warmup_runs, reuse_static=True)

    return compiled
