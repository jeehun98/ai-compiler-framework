from __future__ import annotations
from typing import Dict, Optional

from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda.context import CudaEmitContext
from ..emitters.cuda.gemm import gemm as emit_gemm
from ..emitters.cuda.bias_add import bias_add as emit_bias_add
from ..emitters.cuda.reduce_sum import reduce_sum as emit_reduce_sum

class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, name: str, bias: bool = True):
        super().__init__(name)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bias = bool(bias)

    def emit(self, b, x: int, *, ctx: CudaEmitContext) -> int:
        x_spec = b.values[x].spec
        
        # 출력 형상 결정: (*, in) -> (*, out)
        y_shape = (*x_spec.shape[:-1], self.out_features)
        
        # 1. 출력 Spec 정의 (입력 x의 장치/타입 상속)
        y_spec = TensorSpec(shape=y_shape, dtype=x_spec.dtype, device=x_spec.device)
        y = b.value(f"{self.name}.out", y_spec)
        
        # 2. 가중치(W) 등록
        # [수정] b.dtype과 b.device를 명시하여 'expected None' 에러 방지
        W_spec = TensorSpec(
            shape=(self.out_features, self.in_features), 
            dtype=b.dtype, 
            device=b.device
        )
        W = b.param(f"{self.name}.W", W_spec)

        # y = x @ W^T
        emit_gemm(b, ctx, A=x, B=W, out=y, transA=False, transB=True, name=f"{self.name}.gemm")

        if not self.bias:
            return y

        # 3. Bias 처리
        # [수정] bias 역시 Builder의 메타데이터를 명시적으로 상속
        bias_spec = TensorSpec(
            shape=(self.out_features,), 
            dtype=b.dtype, 
            device=b.device
        )
        bias_val = b.param(f"{self.name}.b", bias_spec)
        
        y2 = b.value(f"{self.name}.out_bias", y_spec)
        emit_bias_add(b, ctx, x=y, bias=bias_val, out=y2, name=f"{self.name}.bias_add", constraints={"inplace_ok": True})
        
        return y2

    def emit_backward(self, b, x: int, W: int, grad_y: int, bias: Optional[int] = None, *, ctx: CudaEmitContext) -> Dict[str, int]:
        """
        Linear 역전파 이미터:
        grad_x = grad_y @ W
        grad_W = grad_y^T @ x
        grad_b = sum(grad_y, axis=0)
        """
        grads = {}

        # 1. d_bias (ReduceSum)
        if bias is not None:
            # 기존 bias의 spec을 복사하여 grad_bias 생성
            g_bias = b.value(f"{self.name}.grad_b", b.values[bias].spec)
            emit_reduce_sum(b, ctx, x=grad_y, out=g_bias, axis=0, name=f"{self.name}.bias_bwd")
            grads["bias"] = g_bias

        # 2. d_W (GEMM: grad_y^T @ x) -> (Out, In)
        g_W = b.value(f"{self.name}.grad_W", b.values[W].spec)
        emit_gemm(b, ctx, A=grad_y, B=x, out=g_W, transA=True, transB=False, name=f"{self.name}.W_bwd")
        grads["weight"] = g_W

        # 3. d_x (GEMM: grad_y @ W) -> (Batch, In)
        g_x = b.value(f"{self.name}.grad_x", b.values[x].spec)
        emit_gemm(b, ctx, A=grad_y, B=W, out=g_x, transA=False, transB=False, name=f"{self.name}.x_bwd")
        grads["input"] = g_x

        return grads