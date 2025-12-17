# examples/python/aicf_fw/backend.py
from __future__ import annotations
from typing import Optional
import os

from .torch_backend import TorchBackend, BackendConfig

# 나중에 AicfBackend 붙일 자리:
# from .aicf_backend import AicfBackend

_GLOBAL_BACKEND = None


def get_backend():
    """
    Backend selector.
    기본은 TorchBackend.
    나중에 AicfBackend 추가되면 환경변수나 config로 스위칭만 하면 됨.
    """
    global _GLOBAL_BACKEND
    if _GLOBAL_BACKEND is None:
        kind = os.environ.get("AICF_BACKEND", "torch").lower()

        if kind == "torch":
            _GLOBAL_BACKEND = TorchBackend(BackendConfig())
        else:
            raise ValueError(f"Unknown backend kind: {kind} (use AICF_BACKEND=torch for now)")

    return _GLOBAL_BACKEND
