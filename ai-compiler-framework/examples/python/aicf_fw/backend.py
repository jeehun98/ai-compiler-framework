# examples/python/aicf_fw/backend.py
from __future__ import annotations
import os

from .torch_backend import TorchBackend, BackendConfig

_GLOBAL_BACKEND = None


def get_backend():
    """
    Backend selector.
    기본은 TorchBackend.
    AICF_BACKEND=aicf 로 설정하면 AicfBackend 사용(지원 안 되는 op는 내부에서 torch로 fallback).
    """
    global _GLOBAL_BACKEND
    if _GLOBAL_BACKEND is None:
        kind = os.environ.get("AICF_BACKEND", "torch").lower()

        if kind == "torch":
            _GLOBAL_BACKEND = TorchBackend(BackendConfig())
        elif kind == "aicf":
            from .aicf_backend import AicfBackend  # lazy import
            _GLOBAL_BACKEND = AicfBackend(BackendConfig())
        else:
            raise ValueError(f"Unknown backend kind: {kind} (use AICF_BACKEND=torch|aicf)")

    return _GLOBAL_BACKEND
