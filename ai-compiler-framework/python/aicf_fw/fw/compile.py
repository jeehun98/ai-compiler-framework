# aicf_fw/fw/compile.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

from aicf_fw.fw.emit_ctx import EmitCtx
from aicf_fw.fw.train_step import CompiledTrainStep
from aicf_fw.fw.naming import opt_m_name, opt_v_name, BC1_NAME, BC2_NAME

from aicf_fw.core_v2.backend_ops import BackendOp, ValueDesc, FrozenGraph
from aicf_fw.core_v2.plan import build_binding_plan, apply_kernel_decisions_stageB
from aicf_fw.core_v2.rewrites.stageC_fuse_epilogue import stageC_fuse_gemm_epilogue
from aicf_fw.core_v2.exec import PlannedExecutor, ExecOptions
from aicf_fw.core_v2.op_attrs.registry import build_op_attr


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    if v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


class _IRShim:
    """
    EmitCtx/FrozenGraph 기반에서도 core_v2 파이프라인이 기대하는 최소 ir 인터페이스.
    - ir.values 만 있으면 stageC/stageB/op_attrs/plan 대부분이 굴러감.
    """
    def __init__(self, *, values: List[ValueDesc], name: str):
        self.values = values
        self.name = name


def _make_params_dict(model) -> Dict[str, torch.Tensor]:
    return {n: t for n, t in model.named_parameters()}


def _lowered_dicts_from_ops(ops: List[BackendOp]) -> List[dict]:
    return [op.to_lowered_dict() for op in ops]


def _alloc_role(values: List[ValueDesc], role: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for vd in values:
        if vd.role != role:
            continue
        out[vd.name] = torch.empty(tuple(vd.shape), device=vd.device, dtype=vd.dtype)
    return out


def _emit_mse_grad(ctx: EmitCtx, y_vid: int, t_vid: int, *, name: str = "mse_grad") -> int:
    y_desc = ctx.values[y_vid]
    dY_vid = ctx.static_vid(name=f"{name}.dY", shape=y_desc.shape)
    ctx.emit_op("mse_grad", inputs=[y_vid, t_vid], outputs=[dY_vid], name=name)
    return dY_vid


def _emit_linear_bwd_like(
    ctx: EmitCtx,
    *,
    x_vid: int,
    W_vid: int,
    dY_vid: int,
    out_prefix: str,
    has_bias: bool,
) -> tuple[int, int, Optional[int]]:
    """
    forward: y = x @ W^T (+b)
    dX = dY @ W
    dW = dY^T @ x
    db = reduce_sum(dY, axis=0)
    """
    x_desc = ctx.values[x_vid]
    dY_desc = ctx.values[dY_vid]

    B = x_desc.shape[0]
    Din = x_desc.shape[1]
    Dout = dY_desc.shape[1]

    # dX
    dX_vid = ctx.static_vid(name=f"{out_prefix}.dX", shape=(B, Din))
    ctx.emit_op(
        "gemm",
        inputs=[dY_vid, W_vid],
        outputs=[dX_vid],
        name=f"{out_prefix}.dx_gemm",
        transA=False,
        transB=False,
    )

    # dW
    dW_vid = ctx.static_vid(name=f"{out_prefix}.dW", shape=(Dout, Din))
    ctx.emit_op(
        "gemm",
        inputs=[dY_vid, x_vid],
        outputs=[dW_vid],
        name=f"{out_prefix}.dW_gemm",
        transA=True,
        transB=False,
    )

    # db
    db_vid: Optional[int] = None
    if has_bias:
        db_vid = ctx.static_vid(name=f"{out_prefix}.db", shape=(Dout,))
        ctx.emit_op(
            "reduce_sum",
            inputs=[dY_vid],
            outputs=[db_vid],
            name=f"{out_prefix}.db_rowsum",
            axis=0,
            keepdim=False,
        )

    return dX_vid, dW_vid, db_vid


def _emit_relu_bwd(ctx: EmitCtx, dY_vid: int, saved_vid: int, *, out_prefix: str) -> int:
    dY_desc = ctx.values[dY_vid]
    dX_vid = ctx.static_vid(name=f"{out_prefix}.dX", shape=dY_desc.shape)
    ctx.emit_op(
        "relu_bwd",
        inputs=[dY_vid, saved_vid],
        outputs=[dX_vid],
        name=f"{out_prefix}.relu_bwd",
    )
    return dX_vid


def _emit_adam_step(
    ctx: EmitCtx,
    *,
    pname: str,
    p_vid: int,
    g_vid: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
):
    """
    adam_step 커널을 호출하는 최소 표현.

    가정(초미니):
      - inputs: [p, g, m, v, bc1_inv, bc2_inv]
      - outputs: [p, m, v] (in-place처럼 동작하지만 명시적으로 outputs로 둠)
      - 하이퍼파라미터는 attrs로 전달
    """
    m_name = opt_m_name(pname)
    v_name = opt_v_name(pname)

    m_vid = ctx.meta_vid(m_name, shape=ctx.values[p_vid].shape, role="meta")
    v_vid = ctx.meta_vid(v_name, shape=ctx.values[p_vid].shape, role="meta")

    bc1_vid = ctx.meta_vid(BC1_NAME, shape=tuple(()), role="meta")
    bc2_vid = ctx.meta_vid(BC2_NAME, shape=tuple(()), role="meta")

    # outputs를 [p, m, v]로 두면 producer가 갱신되고, plan/exec 입장에선 안전함
    ctx.emit_op(
        "adam_step",
        inputs=[p_vid, g_vid, m_vid, v_vid, bc1_vid, bc2_vid],
        outputs=[p_vid, m_vid, v_vid],
        name=f"{pname}.adam_step",
        lr=float(lr),
        beta1=float(beta1),
        beta2=float(beta2),
        eps=float(eps),
        inplace=True,
    )


def compile_train_step(
    model,
    optimizer,
    *,
    B: int,
    D: int,
    device: str,
    dtype: torch.dtype,
    name: str = "train_step",
    warmup_runs: int = 2,
    warmup_inputs: Optional[Dict[str, Any]] = None,
    warmup_required: bool = True,
):
    """
    EmitCtx 기반 "Adam까지" 되는 초미니 train_step 컴파일.

    초미니 전제:
      - 모델이 Sequential(Linear, ReLU, Linear) 패턴
      - ReLU는 saved를 copy_saved로 저장 (relu_bwd에 필요)
      - loss grad는 mse_grad
      - optimizer는 aicf_fw/optim/adam.py Adam (bc1/bc2, m/v)
    """

    dev = torch.device(device) if isinstance(device, str) else device

    # ----------------------------
    # 0) EmitCtx + inputs
    # ----------------------------
    ctx = EmitCtx(B=B, D=D, device=dev, dtype=dtype, name=name)
    x_vid = ctx.input_vid("x", shape=(B, D))
    t_vid = ctx.input_vid("t", shape=(B, D))

    # ----------------------------
    # 1) params pre-register
    # ----------------------------
    params = _make_params_dict(model)
    for pname, pt in params.items():
        ctx.param_vid(pname, shape=tuple(int(x) for x in pt.shape))

    # ----------------------------
    # 2) optimizer state 등록 (meta)
    # ----------------------------
    if hasattr(optimizer, "bind_state_to_ctx"):
        optimizer.bind_state_to_ctx(ctx)

    # ----------------------------
    # 3) forward emit + tape (Linear/ReLU)
    # ----------------------------
    if not hasattr(model, "layers"):
        raise RuntimeError("[compile] expects Sequential-like model with .layers")

    tape_linear: List[Dict[str, Any]] = []
    tape_relu: List[Dict[str, Any]] = []

    cur_vid = x_vid
    for layer in model.layers:
        cname = layer.__class__.__name__.lower()

        if cname == "linear":
            pfx = layer._prefix
            W_name = f"{pfx}.W"
            b_name = f"{pfx}.b"
            has_bias = bool(getattr(layer, "bias", True))

            W_vid = ctx.param_vid(W_name, shape=tuple(int(x) for x in params[W_name].shape))
            b_vid = None
            if has_bias:
                b_vid = ctx.param_vid(b_name, shape=tuple(int(x) for x in params[b_name].shape))

            x_in_vid = cur_vid
            y_vid = layer.emit(ctx, x_in_vid)

            tape_linear.append(
                dict(
                    prefix=pfx,
                    pname_W=W_name,
                    pname_b=b_name,
                    x_in_vid=x_in_vid,
                    y_out_vid=y_vid,
                    W_vid=W_vid,
                    b_vid=b_vid,
                    has_bias=has_bias,
                )
            )
            cur_vid = y_vid
            continue

        if cname == "relu":
            pfx = layer._prefix
            x_in_vid = cur_vid
            y_vid = layer.emit(ctx, x_in_vid)

            # relu.emit이 saved를 "<prefix>.saved"로 만든다고 가정
            saved_vid = ctx.get_vid(f"{pfx}.saved")
            tape_relu.append(dict(prefix=pfx, saved_vid=saved_vid))
            cur_vid = y_vid
            continue

        raise RuntimeError(f"[compile] unsupported layer: {layer.__class__.__name__}")

    y_vid = cur_vid

    # ----------------------------
    # 4) loss grad
    # ----------------------------
    dY_vid = _emit_mse_grad(ctx, y_vid=y_vid, t_vid=t_vid, name="loss_mse_grad")

    # ----------------------------
    # 5) backward (Linear1 <- ReLU <- Linear0)
    # ----------------------------
    if len(tape_linear) == 0:
        raise RuntimeError("[compile] no Linear layers")

    last_lin = tape_linear[-1]
    dX_vid, dW1_vid, db1_vid = _emit_linear_bwd_like(
        ctx,
        x_vid=last_lin["x_in_vid"],
        W_vid=last_lin["W_vid"],
        dY_vid=dY_vid,
        out_prefix=f"{last_lin['prefix']}.bwd",
        has_bias=last_lin["has_bias"],
    )

    if len(tape_relu) > 0:
        last_relu = tape_relu[-1]
        dX_vid = _emit_relu_bwd(ctx, dY_vid=dX_vid, saved_vid=last_relu["saved_vid"], out_prefix=f"{last_relu['prefix']}.bwd")

    dW0_vid = None
    db0_vid = None
    if len(tape_linear) >= 2:
        first_lin = tape_linear[-2]
        _dX0_vid, dW0_vid, db0_vid = _emit_linear_bwd_like(
            ctx,
            x_vid=first_lin["x_in_vid"],
            W_vid=first_lin["W_vid"],
            dY_vid=dX_vid,
            out_prefix=f"{first_lin['prefix']}.bwd",
            has_bias=first_lin["has_bias"],
        )

    # ----------------------------
    # 6) Adam step emit (params update)
    # ----------------------------
    lr = float(getattr(optimizer, "lr", 1e-3))
    beta1 = float(getattr(optimizer, "beta1", 0.9))
    beta2 = float(getattr(optimizer, "beta2", 0.999))
    eps = float(getattr(optimizer, "eps", 1e-8))

    # last linear W/b
    _emit_adam_step(
        ctx,
        pname=last_lin["pname_W"],
        p_vid=last_lin["W_vid"],
        g_vid=dW1_vid,
        lr=lr, beta1=beta1, beta2=beta2, eps=eps,
    )
    if last_lin["has_bias"] and db1_vid is not None and last_lin["b_vid"] is not None:
        _emit_adam_step(
            ctx,
            pname=last_lin["pname_b"],
            p_vid=last_lin["b_vid"],
            g_vid=db1_vid,
            lr=lr, beta1=beta1, beta2=beta2, eps=eps,
        )

    # first linear W/b
    if len(tape_linear) >= 2 and dW0_vid is not None:
        first_lin = tape_linear[-2]
        _emit_adam_step(
            ctx,
            pname=first_lin["pname_W"],
            p_vid=first_lin["W_vid"],
            g_vid=dW0_vid,
            lr=lr, beta1=beta1, beta2=beta2, eps=eps,
        )
        if first_lin["has_bias"] and db0_vid is not None and first_lin["b_vid"] is not None:
            _emit_adam_step(
                ctx,
                pname=first_lin["pname_b"],
                p_vid=first_lin["b_vid"],
                g_vid=db0_vid,
                lr=lr, beta1=beta1, beta2=beta2, eps=eps,
            )

    # ----------------------------
    # 7) Freeze graph -> IR shim -> lowered
    # ----------------------------
    graph = ctx.freeze(meta_as_static=True)  # ✅ meta를 static으로 승격 (plan 호환성)
    ir = _IRShim(values=graph.values, name=name)
    lowered = _lowered_dicts_from_ops(graph.ops)

    # tmp도 capture 안정성을 위해 static으로 승격하고 싶으면 여기서 처리 가능
    for vd in ir.values:
        if vd.role == "tmp":
            vd.role = "static"

    # ----------------------------
    # 8) StageC fuse -> StageB kernel select -> OpAttrs -> Plan -> Exec
    # ----------------------------
    lowered = stageC_fuse_gemm_epilogue(ir, lowered)
    lowered = apply_kernel_decisions_stageB(ir, lowered)

    for i, lop in enumerate(lowered):
        lop_view = dict(lop)
        if "kind" not in lop_view and "op" in lop_view:
            lop_view["kind"] = lop_view["op"]
        _ = build_op_attr(lop_view, ir.values, op_id=i)

    plan = build_binding_plan(ir)

    ex = PlannedExecutor(
        ir=ir,
        lowered=lowered,
        plan=plan,
        opts=ExecOptions(debug=False, require_kernel_id=True),
    )

    # ----------------------------
    # 9) statics allocate + optimizer state 텐서 바인딩 포함
    # ----------------------------
    statics = _alloc_role(ir.values, role="static")

    # optimizer state 텐서(포인터 stable)를 statics에 합친다 (CompiledTrainStep._bind_all에 들어감)
    if hasattr(optimizer, "state_as_statics"):
        statics.update(optimizer.state_as_statics())
    elif hasattr(optimizer, "named_state_tensors"):
        statics.update(optimizer.named_state_tensors())

    # ----------------------------
    # 10) warmup env override + build compiled handle
    # ----------------------------
    warmup_runs_eff = int(warmup_runs)
    if _env_int("AICF_WARMUP", 1) == 0:
        warmup_runs_eff = 0

    compiled = CompiledTrainStep(
        ir=ir,
        lowered=lowered,
        plan=plan,
        ex=ex,
        params=params,
        statics=statics,
        optimizer=optimizer,
        warmup_runs=warmup_runs_eff,
        warmup_required=bool(warmup_required),
    )

    if warmup_runs_eff > 0:
        if warmup_inputs is None and warmup_required:
            raise RuntimeError("warmup_required=True but warmup_inputs is None")
        if warmup_inputs is not None:
            compiled.warmup(warmup_inputs, n=warmup_runs_eff, reuse_static=True)

    return compiled
