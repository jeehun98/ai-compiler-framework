from __future__ import annotations
from typing import TYPE_CHECKING, List

from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda import batchnorm  # 통합 모듈 임포트

if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext

class BatchNorm(Layer):
    def __init__(
        self,
        name: str,
        *,
        eps: float = 1e-5,
        use_running_stats: bool = False,
        affine: bool = True,
    ):
        super().__init__(name)
        self.eps = float(eps)
        self.use_running_stats = bool(use_running_stats)
        self.affine = bool(affine)

    def emit(self, b, x: int, *rest: int, ctx: CudaEmitContext):
        x_spec = b.values[x].spec
        if len(x_spec.shape) != 4:
            raise ValueError(f"BatchNorm expects 4D NCHW; got {x_spec.shape}")
        
        _, C, _, _ = x_spec.shape
        y_spec = TensorSpec(shape=x_spec.shape, dtype=x_spec.dtype, device=x_spec.device)
        y = b.value(f"{self.name}.y", y_spec)

        if not self.use_running_stats:
            # --- Training Mode ---
            if self.affine:
                gamma, beta = rest # [x, gamma, beta]
                ins = [x, gamma, beta]
            else:
                ins = [x]

            # 부산물 Vid 생성
            stat_spec = TensorSpec(shape=(C,), dtype="f32", device=x_spec.device)
            save_mean = b.value(f"{self.name}.save_mean", stat_spec)
            save_rstd = b.value(f"{self.name}.save_rstd", stat_spec)

            batchnorm.emit(
                b, ctx,
                inputs=ins,
                outputs=[y, save_mean, save_rstd],
                eps=self.eps,
                use_running_stats=False,
                name=f"{self.name}.fwd",
            )
            return y, save_mean, save_rstd
        else:
            # --- Inference Mode ---
            # ... (rest 파싱 로직은 기존과 유사하되 batchnorm.emit 호출) ...
            batchnorm.emit(
                b, ctx,
                inputs=list([x] + list(rest)),
                outputs=[y],
                eps=self.eps,
                use_running_stats=True,
                name=f"{self.name}.fwd",
            )
            return y