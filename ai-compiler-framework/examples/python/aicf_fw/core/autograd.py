# aicf_fw/core/autograd.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from .tensor import Tensor

# --- global grad mode ---
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

# --- Node base ---
class Node:
    def __init__(self, inputs: List[Tensor]):
        self.inputs = inputs  # references to input tensors

    def backward(self, out_grad: Tensor) -> List[Optional[Tensor]]:
        """
        Return grads aligned with self.inputs.
        Some inputs may be non-diff -> return None.
        """
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

def _accum_grad(t: Tensor, g: Tensor):
    if not t.requires_grad:
        return
    if t.grad is None:
        t.grad = g
    else:
        from ..backend import get_backend
        backend = get_backend()
        summed = backend.op_call("add", [t.grad.data, g.data], attrs={})
        t.grad = Tensor(summed, requires_grad=False)

def backward(loss: Tensor, grad: Optional[Tensor] = None):
    """
    Reverse-mode autodiff.
    - loss must be scalar (assumed)
    - grad default = 1
    """
    from ..backend import get_backend
    backend = get_backend()

    if grad is None:
        # make scalar 1.0 like loss
        grad = Tensor(backend.ones_like(loss.data), requires_grad=False)

    # seed
    _accum_grad(loss, grad)

    topo = _topo_sort(loss)
    topo.reverse()  # start from loss

    for t in topo:
        if t.creator is None:
            continue
        outg = t.grad
        if outg is None:
            continue
        in_grads = t.creator.backward(outg)
        assert len(in_grads) == len(t.creator.inputs)

        for inp, ig in zip(t.creator.inputs, in_grads):
            if ig is None:
                continue
            _accum_grad(inp, ig)
