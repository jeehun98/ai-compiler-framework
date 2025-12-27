# aicf_fw/core/functional.py
from __future__ import annotations
from typing import Any, Dict
import torch

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

        # -----------------------------------------
        # AICF GEMM contract:
        #  - transA=True  => A is stored as [K,M] contiguous (A_storage = A_logical.t().contiguous())
        #  - transB=True  => B is stored as [N,K] contiguous (B_storage = B_logical.t().contiguous())
        # -----------------------------------------

        # dX = dY @ W^T
        # Need B_storage = W^T stored as [N,K] = [out,in]??  (W logical [in,out] -> W^T logical [out,in])
        # For transB=True, pass B_storage = (W^T).t() == W as [out,in]?  ❌ 헷갈리기 쉬움.
        # 더 안전하게: transB=True이면 "B_storage shape is [N,K]" 이므로
        # B_logical is (K,N). 여기서 W_logical is (K,N)=(in,out).
        # Want B_logical^T in math => use transB=True with B_storage = B_logical.t() = W.t() : [out,in]
        W_storage_for_transB = W.data.t().contiguous()  # [out, in]
        dX_data = backend.op_call(
            "gemm",
            [dY.data, W_storage_for_transB],
            {"transA": False, "transB": True},
        )
        dX = Tensor(dX_data, requires_grad=False)

        # dW = X^T @ dY
        xT = x.data.t().contiguous()        # (in, batch) = (8, 64)
        dW_data = backend.op_call(
            "gemm",
            [xT, dY.data],                  # (8,64) @ (64,8) -> (8,8)
            {"transA": False, "transB": False},
        )
        dW = Tensor(dW_data, requires_grad=False)


        if self.b is not None:
            dB_data = backend.op_call("reduce_sum", [dY.data], {"axis": 0, "keepdim": False})
            dB = Tensor(dB_data, requires_grad=False)
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

        # mean MSE 기준 gradient: (2/numel) * (y - t)
        # out_grad는 스칼라(보통 1)라고 가정하고, 지금은 무시 (안정 우선)
        dY = backend.op_call("mse_grad", [self.y.data, self.t.data], {"scale": 2.0 / self.y.data.numel()})
        return [Tensor(dY, requires_grad=False), None]


# --- ops ---
def linear(x: Tensor, W: Tensor, b: Tensor | None = None) -> Tensor:
    backend = get_backend()

    y_data = backend.op_call("gemm", [x.data, W.data], {"transA": False, "transB": False})
    if b is not None:
        y_data = backend.op_call("bias_add", [y_data, b.data], {})

    y = Tensor(
        y_data,
        requires_grad=(x.requires_grad or W.requires_grad or (b.requires_grad if b else False)),
    )

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
    # torch로 mean loss (스칼라)
    loss_data = ((y.data - t.data) ** 2).mean()
    loss = Tensor(loss_data, requires_grad=y.requires_grad)
    if grad_enabled() and loss.requires_grad:
        loss.creator = MSELossNode(y, t)
    return loss
