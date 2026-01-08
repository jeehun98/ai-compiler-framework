# aicf_fw/core/trace.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Dict, Any
import torch

from .ir import IRGraph, IRValue

_TRACING = False
_IR: Optional[IRGraph] = None

# ---------------------------------------------------------------------
# Tracing value cache (GLOBAL)
#   - cache per-tracing session
#   - key:
#       * AICF Tensor: id(tensor_obj)
#       * torch.Tensor: tensor.data_ptr() (CUDA/CPU 모두 안정적), fallback to id()
# ---------------------------------------------------------------------
_TRACE_VAL_CACHE_OBJ: Dict[int, IRValue] = {}
_TRACE_VAL_CACHE_TORCH: Dict[int, IRValue] = {}


def is_tracing() -> bool:
    return bool(_TRACING)


def get_ir() -> IRGraph:
    if _IR is None:
        raise RuntimeError("IRBuilder is not set. Use `with tracing(ir): ...`")
    return _IR


def trace_reset_cache() -> None:
    """Clear per-session caches. Called automatically by `tracing()`."""
    _TRACE_VAL_CACHE_OBJ.clear()
    _TRACE_VAL_CACHE_TORCH.clear()


def _torch_key(x: torch.Tensor) -> int:
    # Prefer data_ptr when possible (stable identity for storage), fallback to id
    try:
        return int(x.data_ptr())
    except Exception:
        return id(x)


def as_ir_value_obj(obj: Any, *, name: str, shape, dtype, device) -> IRValue:
    """
    Generic object->IRValue mapping (used for AICF Tensor objects).
    shape/dtype/device must be resolvable by caller.
    """
    if not is_tracing():
        raise RuntimeError("as_ir_value_obj() called outside tracing")

    k = id(obj)
    v = _TRACE_VAL_CACHE_OBJ.get(k)
    if v is not None:
        return v

    ir = get_ir()
    v = ir.new_value(name=name, shape=tuple(shape), dtype=str(dtype), device=str(device))
    _TRACE_VAL_CACHE_OBJ[k] = v
    return v


def as_ir_value_torch(x: torch.Tensor, *, name: str) -> IRValue:
    """
    torch.Tensor -> IRValue mapping during tracing.
    Important for optimizer scalars/states (step, bc1_inv, bc2_inv).
    """
    if not is_tracing():
        raise RuntimeError("as_ir_value_torch() called outside tracing")

    k = _torch_key(x)
    v = _TRACE_VAL_CACHE_TORCH.get(k)
    if v is not None:
        return v

    ir = get_ir()
    v = ir.new_value(name=name, shape=tuple(x.shape), dtype=str(x.dtype), device=str(x.device))
    _TRACE_VAL_CACHE_TORCH[k] = v
    return v


@contextmanager
def tracing(ir: IRGraph):
    global _TRACING, _IR
    prev_t = _TRACING
    prev_ir = _IR

    _TRACING = True
    _IR = ir
    trace_reset_cache()

    try:
        yield ir
    finally:
        _TRACING = prev_t
        _IR = prev_ir
        # cache is session-local; clear on exit to avoid accidental cross-session reuse
        trace_reset_cache()
