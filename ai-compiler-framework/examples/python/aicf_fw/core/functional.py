# aicf_fw/core/functional.py
from __future__ import annotations
from typing import Any, Dict
from .tensor import Tensor
from .autograd import Node, grad_enabled
from ..backend import get_backend

# --- Nodes ---
class LinearNode(Node):
    def __init__(self, x: Tensor, W: Tensor, b: Tensor | None):
        inputs = [x, W] + ([b] if b is not None else [])
        super().__init__(inputs)
        self.x = x
        self.W = W
        self.b = b

    def backward(self, out_grad: Tensor):
        backend = get_backend()
        x, W = self.x, self.W

        dY = out_grad

        # dX = dY @ W^T
        dX = Tensor(backend.op_call("gemm", [dY.data, W.data], {"transA": False, "transB": True}), requires_grad=False)

        # dW = X^T @ dY
        dW = Tensor(backend.op_call("gemm", [x.data, dY.data], {"transA": True, "transB": False}), requires_grad=False)

        if self.b is not None:
            # dB = reduce_sum(dY, axis=0)
            dB = Tensor(backend.op_call("reduce_sum", [dY.data], {"axis": 0}), requires_grad=False)
            return [dX, dW, dB]
        return [dX, dW]

class ReLUNode(Node):
    def __init__(self, x: Tensor):
        super().__init__([x])
        self.x = x

    def backward(self, out_grad: Tensor):
        backend = get_backend()
        dX = Tensor(backend.op_call("relu_bwd", [out_grad.data, self.x.data], {}), requires_grad=False)
        return [dX]

class MSELossNode(Node):
    def __init__(self, y: Tensor, t: Tensor):
        super().__init__([y, t])
        self.y = y
        self.t = t

    def backward(self, out_grad: Tensor):
        backend = get_backend()
        # out_grad is scalar multiplier (usually 1)
        dY_base = backend.op_call("mse_grad", [self.y.data, self.t.data], {})
        dY = Tensor(backend.op_call("mul_scalar", [dY_base], {"alpha": out_grad.data}), requires_grad=False) \
             if False else Tensor(dY_base, requires_grad=False)
        # target t usually requires_grad=False, so None is fine
        return [dY, None]

# --- ops ---
def linear(x: Tensor, W: Tensor, b: Tensor | None = None) -> Tensor:
    backend = get_backend()
    y_data = backend.op_call("gemm", [x.data, W.data], {"transA": False, "transB": False})
    if b is not None:
        y_data = backend.op_call("bias_add", [y_data, b.data], {})
    y = Tensor(y_data, requires_grad=(x.requires_grad or W.requires_grad or (b.requires_grad if b else False)))

    if grad_enabled() and y.requires_grad:
        y.creator = LinearNode(x, W, b)
    return y

def relu(x: Tensor) -> Tensor:
    backend = get_backend()
    y_data = backend.op_call("relu", [x.data], {})
    y = Tensor(y_data, requires_grad=x.requires_grad)
    if grad_enabled() and y.requires_grad:
        y.creator = ReLUNode(x)
    return y

def mse_loss(y: Tensor, t: Tensor) -> Tensor:
    backend = get_backend()
    loss_data = backend.op_call("mse", [y.data, t.data], {})
    loss = Tensor(loss_data, requires_grad=y.requires_grad)
    if grad_enabled() and loss.requires_grad:
        loss.creator = MSELossNode(y, t)
    return loss
