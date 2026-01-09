# aicf_fw/core/module.py
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterator, Tuple, Optional, Any, Callable, Sequence, Literal

from .tensor import Tensor
import torch


LossKind = Literal["mse"]


class Module:
    """
    Minimal nn.Module-like base.
    - Registers parameters and child modules.
    - Provides parameters()/named_parameters()/modules()/named_modules()
    - zero_grad() included.

    Added:
    - compile(): compile+warmup+capture+trace into a CompileArtifact and store it.
    - replay(): replay captured CUDA graph for the compiled step.
    - get_artifact(): access last CompileArtifact.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())  # name -> Tensor
        object.__setattr__(self, "_modules", OrderedDict())     # name -> Module
        object.__setattr__(self, "_compiled_artifact", None)    # CompileArtifact | None

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

    # -------------------------
    # Attribute hooks (optional convenience)
    # - Assigning Tensor to attribute auto-registers as parameter
    # - Assigning Module to attribute auto-registers as child
    # -------------------------
    def __setattr__(self, name: str, value):
        if isinstance(value, Tensor):
            # treat as parameter by default
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
        # de-dup by id for shared parameters
        seen = set()

        def _emit(name: str, t: Tensor):
            if not t.requires_grad:
                return
            tid = id(t)
            if tid in seen:
                return
            seen.add(tid)
            yield name, t

        # own params
        for n, p in self._parameters.items():
            full = f"{prefix}.{n}" if prefix else n
            yield from _emit(full, p)

        # child params
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
    # Mode (optional)
    # -------------------------
    def train(self, mode: bool = True):
        for _, m in self._modules.items():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # -------------------------
    # Grad utils
    # -------------------------
    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.parameters(recurse=True):
            if set_to_none:
                p.grad = None
            else:
                # if you want zeros, do it here (requires backend zeros_like)
                p.grad = None

    # -------------------------
    # Call
    # -------------------------
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # -------------------------
    # checkpointing (minimal)
    # -------------------------
    def state_dict(self) -> Dict[str, torch.Tensor]:
        sd: Dict[str, torch.Tensor] = {}
        for name, p in self.named_parameters(recurse=True):
            sd[name] = p.data.detach().clone()
        return sd

    def load_state_dict(self, sd: Dict[str, Any], strict: bool = True) -> None:
        cur = {name: p for name, p in self.named_parameters(recurse=True)}
        missing = []
        unexpected = []

        for k in sd.keys():
            if k not in cur:
                unexpected.append(k)

        for k, p in cur.items():
            if k not in sd:
                missing.append(k)
                continue
            src = sd[k]
            if not isinstance(src, torch.Tensor):
                raise TypeError(f"state_dict[{k}] must be torch.Tensor, got {type(src)}")
            if tuple(src.shape) != tuple(p.data.shape):
                raise ValueError(f"shape mismatch for {k}: {tuple(src.shape)} vs {tuple(p.data.shape)}")
            if src.dtype != p.data.dtype:
                if strict:
                    raise ValueError(f"dtype mismatch for {k}: {src.dtype} vs {p.data.dtype}")
                src = src.to(dtype=p.data.dtype)
            if src.device != p.data.device:
                src = src.to(device=p.data.device)

            # copy into existing tensor (preserves storage for CUDA Graph friendliness)
            p.data.copy_(src)

        if strict and (missing or unexpected):
            raise KeyError(f"load_state_dict strict failed. missing={missing}, unexpected={unexpected}")

    # ============================================================
    # New: compile/replay API (framework-style)
    # ============================================================
    def compile(
        self,
        *,
        # Mode 1: explicit step_fn (advanced / custom)
        step_fn: Optional[Callable[[], None]] = None,

        # Mode 2: auto train step (framework-style)
        optim: Optional[Any] = None,
        x: Optional[Any] = None,
        t: Optional[Any] = None,
        loss: LossKind = "mse",

        # compile/capture options
        name: str = "train_step",
        warmup_runs: int = 2,
        warmup_sync: bool = True,
        validate: bool = True,
        trace: bool = True,
        enforce_ops: Sequence[str] = ("adam_step",),
        torch_sync: bool = True,
    ):
        """
        Compile + validate + lower + warmup + CUDA graph capture + runtime trace.

        Two modes:
          - step_fn is provided: compile exactly that closure.
          - step_fn is None: build a standard train-step with (optim, x, t, loss).

        Returns:
          CompileArtifact (also saved into self._compiled_artifact)
        """
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
        object.__setattr__(self, "_compiled_artifact", art)
        return art

    def replay(self) -> None:
        """
        Replay captured CUDA graph for the last compiled step.
        """
        art = getattr(self, "_compiled_artifact", None)
        if art is None:
            raise RuntimeError("Model is not compiled. Call model.compile(...) first.")
        art.backend.replay()

    def get_artifact(self):
        """
        Get last CompileArtifact.
        """
        art = getattr(self, "_compiled_artifact", None)
        if art is None:
            raise RuntimeError("Model is not compiled. Call model.compile(...) first.")
        return art

    def compile_train(
        self,
        *,
        optim: Any,
        input_spec: Dict[str, Tuple[Tuple[int, ...], torch.dtype, str]],
        loss: str = "mse",
        name: str = "train_step",
        warmup_runs: int = 2,
        warmup_sync: bool = True,
        validate: bool = True,
        trace: bool = True,
        enforce_ops: Sequence[str] = ("adam_step",),
        torch_sync: bool = True,
    ):
        """
        Framework-style training compile:
        - allocates static input buffers from input_spec
        - captures a CUDA graph that reads those buffers
        - returns TrainGraph which can be fed via set_inputs() + replay()
        """
        from aicf_fw.core.train_graph import TrainGraph

        tg = TrainGraph(
            self,
            optim,
            input_spec=input_spec,
            loss=loss,  # type: ignore[arg-type]
            name=name,
            warmup_runs=warmup_runs,
            warmup_sync=warmup_sync,
            validate=validate,
            trace=trace,
            enforce_ops=tuple(enforce_ops),
            torch_sync=torch_sync,
        )
        return tg
