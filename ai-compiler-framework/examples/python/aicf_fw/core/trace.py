# aicf_fw/core/trace.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional
from .ir import IRGraph

_TRACING = False
_IR: Optional[IRGraph] = None


def is_tracing() -> bool:
    return bool(_TRACING)


def get_ir() -> IRGraph:
    if _IR is None:
        raise RuntimeError("IRBuilder is not set. Use `with tracing(ir): ...`")
    return _IR


@contextmanager
def tracing(ir: IRGraph):
    global _TRACING, _IR
    prev_t = _TRACING
    prev_ir = _IR
    _TRACING = True
    _IR = ir
    try:
        yield ir
    finally:
        _TRACING = prev_t
        _IR = prev_ir
