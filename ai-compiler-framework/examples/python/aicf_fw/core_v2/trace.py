from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .ir import IRGraph, IRValue


@dataclass
class _TraceState:
    ir: IRGraph
    obj_cache: Dict[int, IRValue]
    torch_cache: Dict[Tuple[int, Tuple[int, ...], str, str], IRValue]


_STATE: Optional[_TraceState] = None


def is_tracing() -> bool:
    return _STATE is not None


def get_ir() -> IRGraph:
    if _STATE is None:
        raise RuntimeError("core_v2.trace.get_ir(): not tracing")
    return _STATE.ir


def _torch_key(x: torch.Tensor) -> Tuple[int, Tuple[int, ...], str, str]:
    # data_ptr 기반 + shape/dtype/device 포함 (scalar/버퍼 핸들 동일성)
    try:
        ptr = int(x.data_ptr())
    except Exception:
        ptr = id(x)
    return (ptr, tuple(x.shape), str(x.dtype), str(x.device))


def as_ir_value_obj(obj: Any, *, name: str, shape, dtype, device) -> IRValue:
    """
    Python object(예: SymTensor wrapper)의 identity를 IRValue로 안정 매핑.
    """
    if _STATE is None:
        raise RuntimeError("as_ir_value_obj() called outside tracing")
    k = id(obj)
    v = _STATE.obj_cache.get(k)
    if v is not None:
        return v
    v = _STATE.ir.new_value(name=name, shape=tuple(shape), dtype=str(dtype), device=str(device))
    _STATE.obj_cache[k] = v
    return v


def as_ir_value_torch(x: torch.Tensor, *, name: str) -> IRValue:
    """
    torch.Tensor 핸들을 IRValue로 안정 매핑 (step/bc1_inv/bc2_inv 같은 스칼라 상태용).
    """
    if _STATE is None:
        raise RuntimeError("as_ir_value_torch() called outside tracing")
    k = _torch_key(x)
    v = _STATE.torch_cache.get(k)
    if v is not None:
        return v
    v = _STATE.ir.new_value(name=name, shape=tuple(x.shape), dtype=str(x.dtype), device=str(x.device))
    _STATE.torch_cache[k] = v
    return v


@contextmanager
def tracing(ir: IRGraph):
    global _STATE
    prev = _STATE
    _STATE = _TraceState(ir=ir, obj_cache={}, torch_cache={})
    try:
        yield ir
    finally:
        _STATE = prev
