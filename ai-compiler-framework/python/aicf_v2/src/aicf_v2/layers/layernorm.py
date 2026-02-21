from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Union

from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda import layernorm  # 통합된 layernorm 모듈 임포트

if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext

class LayerNorm(Layer):
    """
    Layer Normalization 레이어.
    2D 입력 (M, N)을 지원하며, 학습 시 미분 가능한 부산물(mean, rstd)을 생성합니다.
    """

    def __init__(self, name: str, *, eps: float = 1e-5, affine: bool = True):
        super().__init__(name)
        self.eps = float(eps)
        self.affine = bool(affine)

    def emit(self, b, x: int, *rest: int, ctx: CudaEmitContext) -> Union[int, Tuple[int, int, int]]:
        """
        Forward: LayerNorm 연산을 수행하고 결과를 반환합니다.
        - affine=True:  inputs=[x, gamma, beta]
        - affine=False: inputs=[x]
        """
        x_spec = b.values[x].spec
        if len(x_spec.shape) != 2:
            raise ValueError(f"LayerNorm expects 2D (M, N); got shape={x_spec.shape}")
        
        M, N = x_spec.shape

        # 1. 출력 Spec 정의
        y_spec = TensorSpec(shape=(M, N), dtype=x_spec.dtype, device=x_spec.device)
        # 부산물은 수치 안정성을 위해 fp32[M] 고정
        stat_spec = TensorSpec(shape=(M,), dtype="f32", device=x_spec.device)

        y = b.value(f"{self.name}.y", y_spec)
        mean = b.value(f"{self.name}.mean", stat_spec)
        rstd = b.value(f"{self.name}.rstd", stat_spec)

        # 2. 인자 구성
        if self.affine:
            if len(rest) != 2:
                raise ValueError(f"LayerNorm(affine=True) expects (x, gamma, beta); got {1+len(rest)} args")
            gamma, beta = rest
            ins = [x, gamma, beta]
        else:
            ins = [x]

        # 3. 통합된 layernorm.emit 호출
        # 이제 emit_backward를 직접 구현하지 않아도 Context가 이 노드를 보고 역연산을 생성합니다.
        layernorm.emit(
            b, ctx,
            inputs=ins,
            outputs=[y, mean, rstd],
            eps=self.eps,
            name=f"{self.name}.fwd",
        )

        # 학습 시에는 부산물까지 반환하여 후속 레이어에서 참조 가능하게 함
        return y, mean, rstd