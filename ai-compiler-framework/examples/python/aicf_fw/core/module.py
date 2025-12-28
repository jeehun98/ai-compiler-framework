# aicf_fw/core/module.py
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterator, Tuple, Optional, Any

from .tensor import Tensor
import torch

class Module:
    """
    Minimal nn.Module-like base.
    - Registers parameters and child modules.
    - Provides parameters()/named_parameters()/modules()/named_modules()
    - zero_grad() included.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())  # name -> Tensor
        object.__setattr__(self, "_modules", OrderedDict())     # name -> Module

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
        import torch
        sd: Dict[str, torch.Tensor] = {}

        for name, p in self.named_parameters(recurse=True):
            # store raw torch tensor (detach to be safe)
            sd[name] = p.data.detach().clone()

        return sd

    def load_state_dict(self, sd: Dict[str, Any], strict: bool = True) -> None:
        import torch

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
                # allow dtype mismatch only if you want; default strict
                if strict:
                    raise ValueError(f"dtype mismatch for {k}: {src.dtype} vs {p.data.dtype}")
                src = src.to(dtype=p.data.dtype)
            if src.device != p.data.device:
                src = src.to(device=p.data.device)

            # copy into existing tensor (preserves storage for CUDA Graph friendliness)
            p.data.copy_(src)

        if strict and (missing or unexpected):
            raise KeyError(f"load_state_dict strict failed. missing={missing}, unexpected={unexpected}")
