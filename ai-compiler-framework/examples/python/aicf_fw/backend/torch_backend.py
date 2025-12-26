# aicf_fw/backend/torch_backend.py
from __future__ import annotations
from typing import Any, Dict, List
import torch
from .base import Backend

class TorchBackend(Backend):
    def op_call(self, op: str, inputs: List[Any], attrs: Dict[str, Any]) -> Any:
        if op == "add":
            return inputs[0] + inputs[1]
        if op == "gemm":
            # attrs: transA, transB
            A, B = inputs
            if attrs.get("transA", False): A = A.t()
            if attrs.get("transB", False): B = B.t()
            return A @ B
        if op == "bias_add":
            Y, b = inputs
            return Y + b
        if op == "relu":
            X, = inputs
            return torch.relu(X)
        if op == "relu_bwd":
            # inputs: dY, X
            dY, X = inputs
            return dY * (X > 0).to(dY.dtype)
        if op == "mse":
            # inputs: y, t  -> scalar
            y, t = inputs
            diff = y - t
            return (diff * diff).mean()
        if op == "mse_grad":
            # inputs: y, t -> dY
            y, t = inputs
            # d/dy mean((y-t)^2) = 2*(y-t)/N
            N = y.numel()
            return (2.0 / N) * (y - t)
        if op == "reduce_sum":
            X, = inputs
            axis = attrs["axis"]
            return X.sum(dim=axis)
        if op == "sgd_step":
            # inputs: param, grad
            p, g = inputs
            lr = attrs["lr"]
            return p - lr * g
        raise KeyError(f"Unknown op: {op}")

    def ones_like(self, x: Any) -> Any:
        return torch.ones_like(x)
