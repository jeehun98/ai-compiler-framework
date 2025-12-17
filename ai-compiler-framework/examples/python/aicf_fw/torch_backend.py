# examples/python/aicf_fw/torch_backend.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any, Dict

import torch

from .tensor import Tensor
from .utils.profiling import OpProfiler


@dataclass
class BackendConfig:
    mode: str = "eager"          # "eager" | "bench" | "capture"
    enable_profiler: bool = True


class TorchBackend:
    """
    ✅ Reference backend
    - 목적: correctness / 프레임워크 스켈레톤 / 수렴 확인
    - torch 연산 그대로 사용
    - replay는 미지원(진짜 CUDA graph capture backend는 별도로 붙일 것)
    """
    def __init__(self, cfg: Optional[BackendConfig] = None) -> None:
        self.cfg = cfg or BackendConfig()
        self.profiler = OpProfiler(enabled=self.cfg.enable_profiler)

    def set_mode(self, mode: str) -> None:
        self.cfg.mode = mode

    # capture placeholders
    def warmup(self, model: Any, sample_batch: Any) -> None:
        return

    def capture_begin(self) -> None:
        self.set_mode("capture")
        return

    def capture_end(self) -> None:
        return

    def replay(self) -> None:
        raise NotImplementedError("TorchBackend에서는 replay 미구현")

    # ops
    def gemm(self, a: Tensor, b: Tensor, bias: Optional[Tensor] = None,
             act: Optional[str] = None, attrs: Optional[Dict[str, Any]] = None) -> Tensor:
        sig = self._sig("gemm", a, b, bias=bias, act=act, attrs=attrs)
        with self.profiler.scope("gemm", sig, mode=self.cfg.mode):
            y = a.t @ b.t
            if bias is not None:
                y = y + bias.t
            if act == "relu":
                y = torch.relu(y)
            return Tensor(y)

    def relu(self, x: Tensor) -> Tensor:
        sig = self._sig("relu", x)
        with self.profiler.scope("relu", sig, mode=self.cfg.mode):
            return Tensor(torch.relu(x.t))

    def mse(self, y: Tensor, t: Tensor) -> Tensor:
        sig = self._sig("mse", y, t)
        with self.profiler.scope("mse", sig, mode=self.cfg.mode):
            return Tensor(torch.mean((y.t - t.t) ** 2))

    def _sig(self, op_name: str, *tensors, **kwargs) -> str:
        parts = [op_name] + [tt.signature() for tt in tensors]
        kv = {k: v for k, v in kwargs.items() if v is not None and k != "attrs"}
        if kv:
            parts.append(str(kv))
        return "|".join(parts)
