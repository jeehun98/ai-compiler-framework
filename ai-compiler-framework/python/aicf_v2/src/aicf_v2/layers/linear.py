from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING

from .base import Layer
from ..tensor_spec import TensorSpec
# 통합된 emitter 모듈들을 임포트합니다.
from ..emitters.cuda import gemm, bias_add

# CudaEmitContext는 런타임 의존성을 피하고 타입 힌트용으로만 사용될 경우
# 아래와 같이 TYPE_CHECKING 블록을 사용하거나 상단에 직접 임포트합니다.
if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext

class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, name: str, bias: bool = True):
        super().__init__(name)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bias = bool(bias)

    def emit(self, b, x: int, *, ctx: CudaEmitContext) -> int:
        """
        Forward: y = x @ W^T + b
        통합된 규격(gemm.emit, bias_add.emit)을 호출하여 Builder에 노드를 누적합니다.
        """
        x_spec = b.values[x].spec
        
        # 1. 출력 Spec 결정: (*, in) -> (*, out)
        y_shape = (*x_spec.shape[:-1], self.out_features)
        y_spec = TensorSpec(shape=y_shape, dtype=x_spec.dtype, device=x_spec.device)
        
        # 2. 가중치(W) 등록 (out, in)
        W_spec = TensorSpec(
            shape=(self.out_features, self.in_features), 
            dtype=b.dtype, 
            device=b.device
        )
        W = b.param(f"{self.name}.W", W_spec)

        # 3. 행렬 곱 실행 (y_gemm = x @ W^T)
        y_gemm_vid = b.value(f"{self.name}.out_gemm", y_spec)
        # 이제 gemm.py 내부의 emit() 함수가 호출되어 'gemm' kind의 EmitNode가 생성됩니다.
        gemm.emit(
            b, ctx, 
            A=x, B=W, out=y_gemm_vid, 
            transA=False, transB=True, 
            name=f"{self.name}.gemm"
        )

        if not self.bias:
            return y_gemm_vid

        # 4. Bias 처리 (y_out = y_gemm + b)
        bias_spec = TensorSpec(
            shape=(self.out_features,), 
            dtype=b.dtype, 
            device=b.device
        )
        bias_val = b.param(f"{self.name}.b", bias_spec)
        
        y_out = b.value(f"{self.name}.out", y_spec)
        # bias_add.py 내부의 emit() 함수를 호출합니다.
        bias_add.emit(
            b, ctx, 
            x=y_gemm_vid, bias=bias_val, out=y_out, 
            name=f"{self.name}.bias_add", 
            constraints={"inplace_ok": True}
        )
        
        return y_out