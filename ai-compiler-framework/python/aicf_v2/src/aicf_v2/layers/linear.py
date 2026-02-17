from __future__ import annotations
from typing import Dict, Optional, List

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
        """Forward: y = x @ W^T + b"""
        x_spec = b.values[x].spec
        
        # 출력 형상 결정: (*, in) -> (*, out)
        y_shape = (*x_spec.shape[:-1], self.out_features)
        y_spec = TensorSpec(shape=y_shape, dtype=x_spec.dtype, device=x_spec.device)
        
        # 1. 가중치(W) 등록 (out, in)
        W_spec = TensorSpec(
            shape=(self.out_features, self.in_features), 
            dtype=b.dtype, 
            device=b.device
        )
        W = b.param(f"{self.name}.W", W_spec)

        # 2. 행렬 곱 실행 (y = x @ W^T)
        y = b.value(f"{self.name}.out_gemm", y_spec)
        emit_gemm(b, ctx, A=x, B=W, out=y, transA=False, transB=True, name=f"{self.name}.gemm")

        if not self.bias:
            return y

        # 3. Bias 처리
        bias_spec = TensorSpec(
            shape=(self.out_features,), 
            dtype=b.dtype, 
            device=b.device
        )
        bias_val = b.param(f"{self.name}.b", bias_spec)
        
        y_out = b.value(f"{self.name}.out", y_spec)
        emit_bias_add(b, ctx, x=y, bias=bias_val, out=y_out, 
                      name=f"{self.name}.bias_add", constraints={"inplace_ok": True})
        
        return y_out

    def emit_backward(self, b, ctx: CudaEmitContext, inputs: List[int], outputs: List[int], 
                      grad_y: int, params: List[int], **kwargs) -> Dict[str, int]:
        """
        Linear 역전파 (통합 규격 적용):
        - inputs[0]: x (입력)
        - params[0]: W (가중치)
        - params[1]: b (바이어스, 존재 시)
        - grad_y: 상위에서 전파된 dy
        """
        x = inputs[0]
        W = params[0]
        bias = params[1] if len(params) > 1 else None
        
        grads = {}

        # 1. d_bias (ReduceSum)
        if bias is not None:
            # bias와 동일한 spec으로 grad 생성
            g_bias = b.value(f"{self.name}.grad_b", b.values[bias].spec)
            emit_reduce_sum(b, ctx, x=grad_y, out=g_bias, axis=0, name=f"{self.name}.bias_bwd")
            grads["bias"] = g_bias

        # 2. d_W (GEMM: grad_y^T @ x) -> 형상 (Out, In) 일치 확인
        g_W = b.value(f"{self.name}.grad_W", b.values[W].spec)
        emit_gemm(b, ctx, A=grad_y, B=x, out=g_W, transA=True, transB=False, name=f"{self.name}.W_bwd")
        grads["weight"] = g_W

        # 3. d_x (GEMM: grad_y @ W) -> 형상 (Batch, In)
        # Bwd 연산 시 W는 (Out, In)이므로 grad_y(Batch, Out) @ W(Out, In) -> (Batch, In)
        g_x = b.value(f"{self.name}.grad_x", b.values[x].spec)
        emit_gemm(b, ctx, A=grad_y, B=W, out=g_x, transA=False, transB=False, name=f"{self.name}.x_bwd")
        grads["input"] = g_x

        return grads