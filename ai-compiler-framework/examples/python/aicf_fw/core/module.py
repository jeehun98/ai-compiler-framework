# aicf_fw/core/module.py
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterator, Tuple, Optional, Any, Callable, Sequence, Literal, List

import torch

from .autograd import Tensor


LossKind = Literal["mse"]
InputSpec = Dict[str, Tuple[Tuple[int, ...], torch.dtype, str]]  # name -> (shape, dtype, device)


def _make_static_tensor(shape: Tuple[int, ...], dtype: torch.dtype, device: str, name: str) -> Tensor:
    t = torch.zeros(shape, device=device, dtype=dtype)
    return Tensor(t, requires_grad=False, name=name)


class Module:
    """
    Minimal nn.Module-like base.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())   # name -> Tensor
        object.__setattr__(self, "_modules", OrderedDict())      # name -> Module
        object.__setattr__(self, "_compiled_artifact", None)     # CompileArtifact | None
        object.__setattr__(self, "_static_inputs", None)         # Dict[str, Tensor] | None

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
    # static inputs API (replaces TrainGraph)
    # ============================================================
    def set_inputs(self, **kwargs: torch.Tensor) -> None:
        inputs = getattr(self, "_static_inputs", None)
        if inputs is None:
            raise RuntimeError("No static inputs. Compile with input_spec=... first.")

        for k, v in kwargs.items():
            if k not in inputs:
                raise KeyError(f"Unknown input '{k}'. Known: {list(inputs.keys())}")
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"set_inputs expects torch.Tensor for '{k}', got {type(v)}")

            dst = inputs[k].data
            if tuple(v.shape) != tuple(dst.shape):
                raise ValueError(f"shape mismatch for input '{k}': {tuple(v.shape)} vs {tuple(dst.shape)}")
            if v.dtype != dst.dtype:
                v = v.to(dtype=dst.dtype)
            if v.device != dst.device:
                v = v.to(device=dst.device)

            dst.copy_(v)

    def get_inputs(self) -> Dict[str, Tensor]:
        inputs = getattr(self, "_static_inputs", None)
        if inputs is None:
            raise RuntimeError("No static inputs. Compile with input_spec=... first.")
        return inputs

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
        input_spec: Optional[InputSpec] = None,
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

        # -------------------------
        # 0) build step_fn if needed
        # -------------------------
        if step_fn is None:
            if optim is None:
                raise ValueError("Either provide step_fn=..., or provide optim=... for auto train_step.")
            if loss != "mse":
                raise ValueError(f"Unsupported loss='{loss}' in v0. Only 'mse' is supported.")

            if input_spec is not None:
                inputs: Dict[str, Tensor] = {
                    k: _make_static_tensor(shape, dtype, device, name=k)
                    for k, (shape, dtype, device) in input_spec.items()
                }
                object.__setattr__(self, "_static_inputs", inputs)

                x = inputs.get("x", None)
                t = inputs.get("t", None)
                if x is None or t is None:
                    raise ValueError("input_spec must include keys 'x' and 't' in v0.")

                def _auto_train_step():
                    optim.zero_grad()
                    y = self(x)
                    dY = F.mse_grad(y, t)
                    autograd_backward(y, grad=dY, accumulate=False)
                    optim.step_()

                step_fn = _auto_train_step
            else:
                if x is None or t is None:
                    raise ValueError("Either provide input_spec=..., or provide x=..., t=... for auto train_step.")

                def _auto_train_step():
                    optim.zero_grad()
                    y = self(x)
                    dY = F.mse_grad(y, t)
                    autograd_backward(y, grad=dY, accumulate=False)
                    optim.step_()

                step_fn = _auto_train_step

        # -------------------------
        # 1) compile + capture
        #    - compile.py가 STATIC env(temps/gradpool/saved)를 autobind로 채움
        # -------------------------
        art = compile_and_capture(
            step_fn,
            name=name,
            warmup_runs=warmup_runs,
            warmup_sync=warmup_sync,
            validate=validate,
            trace=trace,
            enforce_ops=tuple(enforce_ops),
            torch_sync=torch_sync,
            autobind_env=True,  # static env
        )

        # -------------------------
        # 2) attach LIVE env provider (params/optim/grads/x/t)
        # -------------------------
        if attach_env:
            art.attach_env_provider(lambda: _build_env_live(art=art, model=self, optim=optim, x=x, t=t))

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


def _unwrap_torch(x) -> torch.Tensor:
    return x.data if hasattr(x, "data") else x


def _build_env_live(*, art, model, optim, x, t) -> Dict[int, torch.Tensor]:
    """
    LIVE vid->torch.Tensor binding만 생성.
    - params / optim state / grads / inputs(x,t)
    - temps/gradpool/saved 등 STATIC은 compile.py autobind_env_from_lowered로 이미 art.env에 있음
    """
    if x is None or t is None:
        raise RuntimeError("_build_env_live: x/t must be provided (or compile with input_spec=... so they exist).")

    ir = art.ir
    lowered = art.lowered
    env: Dict[int, torch.Tensor] = {}

    # 1) bind x/t
    x_t = _unwrap_torch(x)
    t_t = _unwrap_torch(t)

    for vid, val in ir.values.items():
        nm = getattr(val, "name", "")
        if nm == "x":
            env[int(vid)] = x_t
        elif nm == "t":
            env[int(vid)] = t_t

    # 2) bind model params in stable order using Linear node vids
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
        raise RuntimeError(
            f"_build_env_live: param count mismatch. model={len(model_params)} ir_linear_params={len(param_vids)}"
        )

    for vid, p in zip(param_vids, model_params):
        env[int(vid)] = p

    # 3) bind Adam scalar state by IRValue.name
    if optim is not None:
        step_t = getattr(optim, "step", None)
        bc1_t = getattr(optim, "bc1_inv", None)
        bc2_t = getattr(optim, "bc2_inv", None)

        for vid, val in ir.values.items():
            nm = getattr(val, "name", "")
            if nm in ("step", "step_i32", "global_step") and step_t is not None:
                env[int(vid)] = step_t
            elif nm in ("bc1_inv", "bias_corr_out1") and bc1_t is not None:
                env[int(vid)] = bc1_t
            elif nm in ("bc2_inv", "bias_corr_out2") and bc2_t is not None:
                env[int(vid)] = bc2_t

    # 4) bind per-param state + grads by lowered adam_step order
    if optim is not None:
        oparams = getattr(optim, "params", None)
        om = getattr(optim, "m", None)
        ov = getattr(optim, "v", None)
        if oparams is None or om is None or ov is None:
            raise RuntimeError("_build_env_live: optim missing params/m/v")

        adam_items = [it for it in lowered if it["op"] == "adam_step"]
        if len(adam_items) != len(oparams):
            raise RuntimeError(
                f"_build_env_live: adam_step count mismatch. lowered={len(adam_items)} optim.params={len(oparams)}"
            )

        for i, it in enumerate(adam_items):
            in_vids = list(it.get("inputs", []))
            if len(in_vids) < 6:
                raise RuntimeError(f"_build_env_live: bad adam_step inputs at i={i}: {in_vids}")

            p_in, g_in, m_in, v_in = map(int, in_vids[:4])

            gwrap = getattr(oparams[i], "grad", None)
            if gwrap is None:
                raise RuntimeError(
                    f"_build_env_live: optim.params[{i}].grad is None. warmup must materialize grads."
                )
            env[g_in] = gwrap.data
            env[m_in] = om[i].data
            env[v_in] = ov[i].data
            env[p_in] = env[p_in]  # already bound

    return env
