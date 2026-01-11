# aicf_fw/core/module.py
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterator, Tuple, Optional, Any, Callable, Sequence, Literal, List

from .tensor import Tensor
import torch


LossKind = Literal["mse"]


class Module:
    """
    Minimal nn.Module-like base.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())  # name -> Tensor
        object.__setattr__(self, "_modules", OrderedDict())     # name -> Module
        object.__setattr__(self, "_compiled_artifact", None)

    # -------------------------
    # Registration
    # -------------------------
    def register_parameter(self, name: str, param: Optional[Tensor]) -> None:
        if param is None:
            return
        if not isinstance(param, Tensor):
            raise TypeError(f"parameter '{name}' must be a Tensor, got {type(param)}")
        self._parameters[name] = param

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        if module is None:
            return
        if not isinstance(module, Module):
            raise TypeError(f"module '{name}' must be a Module, got {type(module)}")
        self._modules[name] = module

    def __setattr__(self, name: str, value):
        if isinstance(value, Tensor):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        object.__setattr__(self, name, value)

    # -------------------------
    # Iterators
    # -------------------------
    def parameters(self, recurse: bool = True) -> Iterator[Tensor]:
        for _, p in self.named_parameters(recurse=recurse):
            yield p

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        seen = set()

        def _emit(name: str, t: Tensor):
            if not t.requires_grad:
                return
            tid = id(t)
            if tid in seen:
                return
            seen.add(tid)
            yield name, t

        for n, p in self._parameters.items():
            full = f"{prefix}.{n}" if prefix else n
            yield from _emit(full, p)

        if recurse:
            for mn, m in self._modules.items():
                child_prefix = f"{prefix}.{mn}" if prefix else mn
                yield from m.named_parameters(prefix=child_prefix, recurse=True)

    def modules(self) -> Iterator["Module"]:
        for _, m in self.named_modules():
            yield m

    def named_modules(self, prefix: str = "") -> Iterator[Tuple[str, "Module"]]:
        yield prefix, self
        for n, m in self._modules.items():
            child_prefix = f"{prefix}.{n}" if prefix else n
            yield from m.named_modules(prefix=child_prefix)

    # -------------------------
    # Grad utils
    # -------------------------
    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.parameters(recurse=True):
            p.grad = None if set_to_none else None

    # -------------------------
    # Call
    # -------------------------
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ============================================================
    # compile/replay API
    # ============================================================
    def compile(
        self,
        *,
        step_fn: Optional[Callable[[], None]] = None,
        optim: Optional[Any] = None,
        x: Optional[Any] = None,
        t: Optional[Any] = None,
        loss: LossKind = "mse",
        name: str = "train_step",
        warmup_runs: int = 2,
        warmup_sync: bool = True,
        validate: bool = True,
        trace: bool = True,
        enforce_ops: Sequence[str] = ("adam_step",),
        torch_sync: bool = True,
        attach_env: bool = True,
    ):
        from aicf_fw.core.compile import compile_and_capture
        from aicf_fw.core.autograd import backward as autograd_backward
        from aicf_fw.core import functional as F

        if step_fn is None:
            if optim is None or x is None or t is None:
                raise ValueError("Either provide step_fn=..., or provide optim=..., x=..., t=... for auto train_step.")
            if loss != "mse":
                raise ValueError(f"Unsupported loss='{loss}' in v0. Only 'mse' is supported.")

            def _auto_train_step():
                optim.zero_grad()
                y = self(x)
                dY = F.mse_grad(y, t)
                autograd_backward(y, grad=dY, accumulate=False)
                optim.step_()

            step_fn = _auto_train_step

        art = compile_and_capture(
            step_fn,
            name=name,
            warmup_runs=warmup_runs,
            warmup_sync=warmup_sync,
            validate=validate,
            trace=trace,
            enforce_ops=tuple(enforce_ops),
            torch_sync=torch_sync,
        )

        if attach_env:
            env = _build_env_exact(art, model=self, optim=optim, x=x, t=t)
            art.attach_env(env)

        object.__setattr__(self, "_compiled_artifact", art)
        return art

    def replay(self) -> None:
        art = getattr(self, "_compiled_artifact", None)
        if art is None:
            raise RuntimeError("Model is not compiled. Call model.compile(...) first.")
        art.backend.replay()

    def get_artifact(self):
        art = getattr(self, "_compiled_artifact", None)
        if art is None:
            raise RuntimeError("Model is not compiled. Call model.compile(...) first.")
        return art


def _dtype_from_ir_dtype_str(s: str) -> torch.dtype:
    ss = str(s)
    if "float32" in ss or "f32" in ss:
        return torch.float32
    if "float16" in ss or "f16" in ss:
        return torch.float16
    if "bfloat16" in ss or "bf16" in ss:
        return torch.bfloat16
    if "int32" in ss or "i32" in ss:
        return torch.int32
    if "int64" in ss or "i64" in ss:
        return torch.int64
    return torch.float32

# aicf_fw/core/module.py 안의 _build_env_exact 교체
from typing import Dict, List, Tuple, Any
import torch


def _dtype_from_ir_dtype_str(s: str) -> torch.dtype:
    ss = str(s)
    if "float32" in ss or "f32" in ss:
        return torch.float32
    if "float16" in ss or "f16" in ss:
        return torch.float16
    if "bfloat16" in ss or "bf16" in ss:
        return torch.bfloat16
    if "int32" in ss or "i32" in ss:
        return torch.int32
    if "int64" in ss or "i64" in ss:
        return torch.int64
    return torch.float32


def _build_env_exact(art, *, model, optim, x, t) -> Dict[int, torch.Tensor]:
    """
    Exact vid->torch.Tensor binding for IRExecutor.

    Rules:
      - Bind x/t by IRValue.name == 'x'/'t'
      - Bind model params by Linear node param vids -> model.named_parameters() order
      - Bind Adam state by param index:
          step, bc1_inv, bc2_inv come from optim
          m/v come from optim.m[i].data / optim.v[i].data
          grad comes from optim.params[i].grad (must exist after warmup)
      - Allocate intermediates needed by lowered ops
      - Apply SSA alias rules (bias_add/step_inc/copy), but adam_step in-place is enforced in IRExecutor too.
    """
    ir = art.ir
    lowered = art.lowered
    env: Dict[int, torch.Tensor] = {}

    def alloc_for_vid(vid: int) -> torch.Tensor:
        v = ir.values[int(vid)]
        device = torch.device(str(v.device))
        dtype = _dtype_from_ir_dtype_str(str(v.dtype))
        shape = tuple(v.shape)
        return torch.empty(shape, device=device, dtype=dtype)

    # -------------------------
    # 1) bind x / t
    # -------------------------
    x_t = x.data if hasattr(x, "data") else x
    t_t = t.data if hasattr(t, "data") else t

    for vid, val in ir.values.items():
        nm = getattr(val, "name", "")
        if nm == "x":
            env[int(vid)] = x_t
        elif nm == "t":
            env[int(vid)] = t_t

    # -------------------------
    # 2) bind model params via Linear node vids (order-stable)
    # -------------------------
    # Collect param vids in execution order: W0,b0,W1,b1,...
    param_vids: List[int] = []
    for n in ir.nodes:
        if n.op != "Linear":
            continue
        W_vid = int(n.inputs[1])
        if W_vid not in param_vids:
            param_vids.append(W_vid)
        if bool(n.attrs.get("bias", False)):
            b_vid = int(n.inputs[2])
            if b_vid not in param_vids:
                param_vids.append(b_vid)

    model_params: List[torch.Tensor] = [p.data for _, p in list(model.named_parameters())]
    if len(model_params) != len(param_vids):
        # not fatal, but it means your param discovery assumption broke
        raise RuntimeError(f"_build_env_exact: param count mismatch. model={len(model_params)} ir_linear_params={len(param_vids)}")

    for vid, p in zip(param_vids, model_params):
        env[int(vid)] = p

    # -------------------------
    # 3) bind Adam scalar state (step, bc1_inv, bc2_inv) EXACT
    # -------------------------
    if optim is not None:
        # These are torch tensors (not wrappers) in your Adam
        # step: int32 scalar
        # bc1_inv/bc2_inv: float32 scalars
        step_t = getattr(optim, "step", None)
        bc1_t = getattr(optim, "bc1_inv", None)
        bc2_t = getattr(optim, "bc2_inv", None)

        # Bind by IRValue.name where possible
        for vid, val in ir.values.items():
            nm = getattr(val, "name", "")
            if nm in ("step", "step_i32", "global_step") and step_t is not None:
                env[int(vid)] = step_t
            elif nm in ("bc1_inv", "bias_corr_out1") and bc1_t is not None:
                env[int(vid)] = bc1_t
            elif nm in ("bc2_inv", "bias_corr_out2") and bc2_t is not None:
                env[int(vid)] = bc2_t

    # -------------------------
    # 4) bind Adam per-param state + grads EXACT by param index
    # -------------------------
    if optim is not None:
        # optim.params is list of aicf Tensor (with .data and .grad)
        oparams = getattr(optim, "params", None)
        om = getattr(optim, "m", None)
        ov = getattr(optim, "v", None)
        if oparams is None or om is None or ov is None:
            raise RuntimeError("_build_env_exact: optim is missing params/m/v required for exact binding")

        # Build a mapping for each adam_step op instance in lowered:
        # We assume lowered optim slice is in param order (it is in your lowering loop).
        adam_items = [it for it in lowered if it["op"] == "adam_step"]
        if len(adam_items) != len(oparams):
            raise RuntimeError(f"_build_env_exact: adam_step count mismatch. lowered={len(adam_items)} optim.params={len(oparams)}")

        for i, it in enumerate(adam_items):
            in_vids = list(it.get("inputs", []))
            # inputs: [p_in, g_in, m_in, v_in, bc1, bc2]
            if len(in_vids) < 6:
                raise RuntimeError(f"_build_env_exact: bad adam_step inputs at i={i}: {in_vids}")

            p_in, g_in, m_in, v_in = map(int, in_vids[:4])

            # p: from model param binding (already set)
            # grad: from oparams[i].grad (aicf Tensor wrapper)
            gwrap = getattr(oparams[i], "grad", None)
            if gwrap is None:
                raise RuntimeError(
                    f"_build_env_exact: optim.params[{i}].grad is None. "
                    "Warmup must materialize all grad buffers before IRExecutor compare."
                )
            env[g_in] = gwrap.data

            # m/v: from optim.m[i] / optim.v[i] (aicf Tensor wrapper)
            env[m_in] = om[i].data
            env[v_in] = ov[i].data

            # NOTE: p_in already bound to param tensor, but we can sanity overwrite:
            env[p_in] = env[p_in]

    # -------------------------
    # 5) allocate remaining tensors needed by lowered ops
    # -------------------------
    for it in lowered:
        for iv in it.get("inputs", []):
            iv = int(iv)
            if iv not in env:
                env[iv] = alloc_for_vid(iv)
        for ovv in it.get("outputs", []):
            ovv = int(ovv)
            if ovv not in env:
                env[ovv] = alloc_for_vid(ovv)

    # -------------------------
    # 6) SSA alias rules for obvious in-place ops
    # -------------------------
    for it in lowered:
        op = it["op"]
        ins = list(it.get("inputs", []))
        outs = list(it.get("outputs", []))

        if op in ("bias_add", "step_inc", "copy"):
            if ins and outs:
                env[int(outs[0])] = env[int(ins[0])]

        # adam_step in-place alias는 executor에서 강제하므로 여기선 굳이 안 해도 됨
        # (해도 무방하지만, executor 강제가 더 안전)

    return env
