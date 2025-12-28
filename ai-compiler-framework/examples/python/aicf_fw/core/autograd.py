from __future__ import annotations

from typing import List, Optional, Dict
from .tensor import Tensor
import torch

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

    ENFORCEMENT:
      - accumulate=True may create new tensors for non-leaf grads (gmap add path).
      - therefore accumulate=True is FORBIDDEN during capture.
    """
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
