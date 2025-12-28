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

        x = self.x          # (B, I)
        W = self.W          # (I, O)
        dY = out_grad       # (B, O)

        # dX = dY @ W^T
        # (B,O) @ (O,I) -> (B,I) == gemm(dY, W, transB=True)
        dX_data = backend.op_call(
            "gemm",
            [dY.data, W.data],
            {"transA": False, "transB": True},
        )
        dX = Tensor(dX_data, requires_grad=False)

        # dW = X^T @ dY
        # (I,B) @ (B,O) -> (I,O) == gemm(x, dY, transA=True)
        dW_data = backend.op_call(
            "gemm",
            [x.data, dY.data],
            {"transA": True, "transB": False},
        )
        dW = Tensor(dW_data, requires_grad=False)

        if self.b is not None:
            # dB = reduce_sum(dY, axis=0)
            dB_data = backend.op_call(
                "reduce_sum",
                [dY.data],
                {"axis": 0, "keepdim": False},
            )
            dB = Tensor(dB_data, requires_grad=False)
            return [dX, dW, dB]

        return [dX, dW]


class ReLUNode(Node):
    def __init__(self, x: Tensor):
        super().__init__([x])
        self.x = x

    def backward(self, out_grad: Tensor):
        backend = get_backend()
        dX_data = backend.op_call("relu_bwd", [out_grad.data, self.x.data], {})
        dX = Tensor(dX_data, requires_grad=False)
        return [dX]


class MSELossNode(Node):
    def __init__(self, y: Tensor, t: Tensor):
        super().__init__([y, t])
        self.y = y
        self.t = t

    def backward(self, out_grad: Tensor):
        backend = get_backend()

        # Your AICF has mse_grad kernel. It already uses scale = 2/numel by default unless attr "scale".
        # out_grad is usually scalar(1). We apply it safely in torch to avoid needing a "scale" op.
        dY_base = backend.op_call("mse_grad", [self.y.data, self.t.data], {})
        # Multiply by out_grad (scalar) in torch to avoid extra custom op.
        # out_grad.data may be a 0-d tensor on cuda/cpu. Normalize it to CUDA scalar.
        og = out_grad.data
        if hasattr(og, "numel") and og.numel() == 1:
            # ensure CUDA
            if og.is_cuda != self.y.data.is_cuda:
                og = og.to(device=self.y.data.device)
        dY_data = dY_base * og
        dY = Tensor(dY_data, requires_grad=False)
        return [dY, None]


# --- ops ---
def linear(x: Tensor, W: Tensor, b: Tensor | None = None) -> Tensor:
    backend = get_backend()

    # y = x @ W
    y_data = backend.op_call("gemm", [x.data, W.data], {"transA": False, "transB": False})

    if b is not None:
        y_data = backend.op_call("bias_add", [y_data, b.data], {})

    req = (x.requires_grad or W.requires_grad or (b.requires_grad if b is not None else False))
    y = Tensor(y_data, requires_grad=req)

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

    # NOTE:
    # AICF currently doesn't expose "mse" forward in OpKind.
    # We'll compute loss forward in torch for now, but keep mse_grad for backward.
    # loss = mean((y - t)^2)
    diff = y.data - t.data
    loss_data = (diff * diff).mean()

    loss = Tensor(loss_data, requires_grad=y.requires_grad)
    if grad_enabled() and loss.requires_grad:
        loss.creator = MSELossNode(y, t)
    return loss
