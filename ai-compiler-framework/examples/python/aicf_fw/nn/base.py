from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import torch

from aicf_fw.fw.naming import param_name


@dataclass
class CompileConfig:
    name: str = "train_step"
    warmup_runs: int = 2
    warmup_inputs: Optional[dict[str, Any]] = None
    warmup_required: bool = True


class Module:
    def __init__(self):
        self._modules: OrderedDict[str, Module] = OrderedDict()
        self._params: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._prefix: str = ""  # set by parent

        # ---- compile/runtime state (fw-style UX) ----
        self._compiled: Any = None                 # compiled train step handle
        self._compile_cfg: Optional[CompileConfig] = None

    # ----- registration -----
    def add_module(self, name: str, m: "Module"):
        self._modules[name] = m
        m._prefix = name if self._prefix == "" else f"{self._prefix}.{name}"

    def register_parameter(self, local_name: str, t: torch.Tensor):
        self._params[local_name] = t

    # ----- query -----
    def named_parameters(self, prefix: str | None = None):
        pfx = self._prefix if prefix is None else prefix

        for local, t in self._params.items():
            yield param_name(pfx, local), t

        for child_name, m in self._modules.items():
            child_prefix = child_name if pfx == "" else f"{pfx}.{child_name}"
            yield from m.named_parameters(prefix=child_prefix)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    # ----- device/dtype -----
    def to(self, device: str | torch.device, dtype: torch.dtype | None = None):
        dev = torch.device(device) if isinstance(device, str) else device
        for k, t in list(self._params.items()):
            self._params[k] = t.to(device=dev, dtype=(dtype if dtype is not None else t.dtype))
        for _, m in self._modules.items():
            m.to(dev, dtype=dtype)
        return self

    # ----- fw-style compile API -----
    def compile(
        self,
        *,
        optimizer,
        B: int,
        D: int,
        device: str,
        dtype: torch.dtype,
        name: str = "train_step",
        warmup_runs: int = 2,
        warmup_inputs: Optional[dict[str, Any]] = None,
        warmup_required: bool = True,
    ):
        """
        Compile a training step and attach it to this module.
        Usage:
            model.compile(optimizer=opt, B=..., D=..., device="cuda:0", dtype=torch.float32, warmup_inputs=...)
        """
        from aicf_fw.fw.compile import compile_train_step

        self._compile_cfg = CompileConfig(
            name=name,
            warmup_runs=warmup_runs,
            warmup_inputs=warmup_inputs,
            warmup_required=warmup_required,
        )

        self._compiled = compile_train_step(
            self,
            optimizer,
            B=B,
            D=D,
            device=device,
            dtype=dtype,
            name=name,
            warmup_runs=warmup_runs,
            warmup_inputs=warmup_inputs,
            warmup_required=warmup_required,
        )
        return self

    def is_compiled(self) -> bool:
        return self._compiled is not None

    def train_step(self, batch: dict[str, Any]):
        if self._compiled is None:
            raise RuntimeError("model is not compiled. call model.compile(...) first.")
        return self._compiled.train_step(batch)

    def capture(self, batch: dict[str, Any]):
        if self._compiled is None:
            raise RuntimeError("model is not compiled. call model.compile(...) first.")
        return self._compiled.capture(batch)

    def replay(self, n: int = 1):
        if self._compiled is None:
            raise RuntimeError("model is not compiled. call model.compile(...) first.")
        return self._compiled.replay(n=n)

    def reset(self):
        if self._compiled is None:
            return
        return self._compiled.reset()

    # ----- IR forward -----
    def forward_ir(self, x_sym, psym: dict[str, object]):
        raise NotImplementedError
