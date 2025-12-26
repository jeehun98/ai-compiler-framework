# aicf_fw/backend/__init__.py
from __future__ import annotations
from typing import Optional
from .base import Backend

_BACKEND: Optional[Backend] = None

def set_backend(b: Backend):
    global _BACKEND
    _BACKEND = b

def get_backend() -> Backend:
    assert _BACKEND is not None, "Backend not set. Call aicf_fw.backend.set_backend(...)"
    return _BACKEND
