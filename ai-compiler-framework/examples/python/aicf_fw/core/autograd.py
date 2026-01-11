# aicf_fw/core/autograd.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


# ============================================================
# Tensor (+ Parameter) lives here now (tensor.py absorbed)
# ============================================================

@dataclass
class TensorMeta:
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


class Tensor:
    """
    data: backend handle (torch.Tensor or None for symbolic)
    creator: Node that created this Tensor (None for leaf/Parameter)
    grad: Tensor or backend handle (we keep as Tensor for simplicity)
    meta: TensorMeta (always available; for symbolic tensors meta is required)
    """
    __slots__ = ("data", "requires_grad", "grad", "creator", "name", "meta")

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        creator=None,
        name: str = "",
        meta: Optional[TensorMeta] = None,
    ):
        self.data = data
        self.requires_grad = requires_grad
        self.grad: Optional["Tensor"] = None
        self.creator = creator
        self.name = name

        if meta is not None:
            self.meta = meta
        else:
            if isinstance(data, torch.Tensor):
                self.meta = TensorMeta(tuple(data.shape), data.dtype, data.device)
            else:
                raise TypeError("Tensor(meta=None) requires torch.Tensor data; for symbolic Tensor, pass meta.")

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.meta.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self.meta.dtype

    @property
    def device(self) -> torch.device:
        return self.meta.device

    @property
    def is_symbolic(self) -> bool:
        return self.data is None

    def zero_grad(self):
        self.grad = None

    def backward(self, grad: Optional["Tensor"] = None):
        backward(self, grad)


class Parameter(Tensor):
    def __init__(self, data: Any, requires_grad: bool = True, name: str = ""):
        super().__init__(data, requires_grad=requires_grad, creator=None, name=name)


# ============================================================
# Grad enable switch (framework-level)
# ============================================================

_grad_enabled = True


class no_grad:
    def __enter__(self):
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = False

    def __exit__(self, exc_type, exc, tb):
        global _grad_enabled
        _grad_enabled = self.prev


def grad_enabled() -> bool:
    return _grad_enabled


# ============================================================
# Capture guard (manually toggled)
# ============================================================

_IN_CAPTURE_GUARD = False


def _set_capture_guard(flag: bool) -> None:
    global _IN_CAPTURE_GUARD
    _IN_CAPTURE_GUARD = bool(flag)


def in_capture() -> bool:
    return _IN_CAPTURE_GUARD


# ============================================================
# Tracing guard (IR compile)
# ============================================================

from aicf_fw.core.compile import is_tracing, get_ir, as_ir_value_obj


# ============================================================
# Node base
# ============================================================

class Node:
    def __init__(self, inputs: List[Tensor]):
        self.inputs = inputs

    def backward(self, out_grad: Tensor) -> List[Optional[Tensor]]:
        raise NotImplementedError


def _topo_sort(root: Tensor) -> List[Tensor]:
    visited = set()
    order: List[Tensor] = []

    def dfs(t: Tensor):
        if id(t) in visited:
            return
        visited.add(id(t))
        if t.creator is not None:
            for inp in t.creator.inputs:
                dfs(inp)
        order.append(t)

    dfs(root)
    return order  # inputs first, root last


# ============================================================
# Leaf grad buffer policy
# ============================================================

def _ensure_leaf_grad_buffer_like(g: torch.Tensor) -> torch.Tensor:
    # MUST be outside capture
    if _IN_CAPTURE_GUARD:
        raise RuntimeError(
            "autograd: allocating leaf grad buffer during capture. "
            "Run warmup backward BEFORE capture to materialize all parameter.grad buffers."
        )
    return torch.empty_like(g)


def _leaf_overwrite(t: Tensor, g: Tensor):
    """
    Overwrite leaf grad in-place using AICF copy:
      grad_buf := g
    """
    if not t.requires_grad:
        return

    from ..backend import get_backend
    backend = get_backend()

    if t.grad is None:
        if _IN_CAPTURE_GUARD:
            raise RuntimeError(
                "autograd(overwrite): leaf.grad is None during capture. "
                "Warmup must materialize ALL parameter.grad buffers and you must not set them to None."
            )
        if g.data is None:
            raise RuntimeError("autograd(overwrite): g.data is None (symbolic). This should not happen outside tracing.")
        buf = _ensure_leaf_grad_buffer_like(g.data)
        t.grad = Tensor(buf, requires_grad=False)

    # pointer-stable overwrite
    backend.op_call_out("copy", [g.data], [t.grad.data], attrs={})


def _leaf_add(t: Tensor, g: Tensor):
    """
    Accumulate on leaf in-place:
      grad_buf += g
    """
    if not t.requires_grad:
        return

    from ..backend import get_backend
    backend = get_backend()

    if t.grad is None:
        if _IN_CAPTURE_GUARD:
            raise RuntimeError("autograd(add): leaf grad materialization attempted during capture.")
        if g.data is None:
            raise RuntimeError("autograd(add): g.data is None (symbolic). This should not happen outside tracing.")
        buf = _ensure_leaf_grad_buffer_like(g.data)
        buf.copy_(g.data)  # outside capture only
        t.grad = Tensor(buf, requires_grad=False)
        return

    backend.op_call_out("add", [t.grad.data, g.data], [t.grad.data], attrs={})


# ============================================================
# Backward
# ============================================================

def backward(loss: Tensor, grad: Optional[Tensor] = None, *, accumulate: bool = False):
    """
    Reverse-mode autodiff.

    Capture-safe policy:
      - Persist grads ONLY on leaf tensors (parameters).
      - Non-leaf grads are stored in a local dict (gmap).
      - overwrite mode uses copy() to keep leaf grad pointer stable.

    TRACING policy:
      - During IR compile/tracing, we do NOT run actual autodiff.
      - We emit an IR node "Backward" that correctly references *existing* traced values.
    """
    # --------------------------------------------------------
    # TRACING PATH (compile/IR)
    # --------------------------------------------------------
    if is_tracing():
        ir = get_ir()

        # IMPORTANT:
        # Do NOT create fresh IRValues here.
        # Use object->IRValue cache so Backward connects to the same value IDs
        # produced by forward ops (Linear/ReLU/MseGrad).
        lv = as_ir_value_obj(
            loss,
            name=loss.name or "loss",
            shape=loss.shape,
            dtype=loss.dtype,
            device=loss.device,
        )
        ins = [lv]

        if grad is not None:
            gv = as_ir_value_obj(
                grad,
                name=grad.name or "dLoss",
                shape=grad.shape,
                dtype=grad.dtype,
                device=grad.device,
            )
            ins.append(gv)

        ir.emit(op="Backward", inputs=ins, outputs=[], attrs={"accumulate": bool(accumulate)})
        return

    # --------------------------------------------------------
    # EXECUTION PATH
    # --------------------------------------------------------
    from ..backend import get_backend
    backend = get_backend()

    if _IN_CAPTURE_GUARD and accumulate:
        raise RuntimeError(
            "autograd.backward(accumulate=True) is forbidden during capture. "
            "Use accumulate=False for capture-safe training replay, or implement out-buffer "
            "non-leaf grad accumulation with a buffer pool."
        )

    if grad is None:
        if _IN_CAPTURE_GUARD:
            raise RuntimeError(
                "autograd.backward: grad=None requires ones_like allocation; "
                "provide explicit grad inside capture."
            )
        grad = Tensor(backend.ones_like(loss.data), requires_grad=False)

    leaf_write = _leaf_add if accumulate else _leaf_overwrite

    # local grad map for non-leaf tensors
    gmap: Dict[int, Tensor] = {id(loss): grad}

    topo = _topo_sort(loss)
    topo.reverse()

    for t in topo:
        if t.creator is None:
            gt = gmap.get(id(t))
            if gt is not None:
                leaf_write(t, gt)
            continue

        outg = gmap.get(id(t))
        if outg is None:
            continue

        in_grads = t.creator.backward(outg)
        assert len(in_grads) == len(t.creator.inputs)

        for inp, ig in zip(t.creator.inputs, in_grads):
            if ig is None:
                continue

            if inp.creator is None:
                leaf_write(inp, ig)
            else:
                if accumulate and (id(inp) in gmap):
                    # This path allocates a new tensor via backend.op_call (NOT capture-safe).
                    # It is safe outside capture and still useful for debugging.
                    summed = backend.op_call("add", [gmap[id(inp)].data, ig.data], attrs={})
                    gmap[id(inp)] = Tensor(summed, requires_grad=False)
                else:
                    gmap[id(inp)] = ig


__all__ = [
    "TensorMeta",
    "Tensor",
    "Parameter",
    "Node",
    "no_grad",
    "grad_enabled",
    "backward",
    "in_capture",
    "_set_capture_guard",
]
