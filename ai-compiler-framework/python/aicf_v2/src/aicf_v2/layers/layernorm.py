from __future__ import annotations

from .base import Layer
from ..tensor_spec import TensorSpec

from ..emitters.cuda.context import CudaEmitContext
from ..emitters.cuda.layernorm_fwd import layernorm_fwd as emit_layernorm_fwd
from ..emitters.cuda.layernorm_bwd import layernorm_bwd as emit_layernorm_bwd


class LayerNormFwd(Layer):
    """
    LayerNorm forward (2D only): x shape (M, N)

    affine=False:
      inputs : [x]
      outputs: [y, mean(fp32[M]), rstd(fp32[M])]

    affine=True:
      inputs : [x, gamma(N), beta(N)]
      outputs: [y, mean, rstd]

    attrs schema: 'LNEP'
    blob       : <f (eps)
    """

    def __init__(self, name: str, *, eps: float = 1e-5, affine: bool = False):
        super().__init__(name)
        self.eps = float(eps)
        self.affine = bool(affine)

    def emit(self, b, x: int, *rest: int, ctx: CudaEmitContext):
        x_spec = b.values[x].spec
        if len(x_spec.shape) != 2:
            raise ValueError(f"LayerNormFwd expects 2D (M,N); got shape={x_spec.shape}")
        M, N = x_spec.shape

        y = b.value(f"{self.name}.y", TensorSpec(shape=(M, N), dtype=x_spec.dtype, device=x_spec.device))
        mean = b.value(f"{self.name}.mean", TensorSpec(shape=(M,), dtype="f32", device=x_spec.device))
        rstd = b.value(f"{self.name}.rstd", TensorSpec(shape=(M,), dtype="f32", device=x_spec.device))

        if self.affine:
            if len(rest) != 2:
                raise ValueError(f"LayerNormFwd(affine) expects (x,gamma,beta); got {1+len(rest)} args")
            gamma, beta = rest
            g_spec = b.values[gamma].spec
            be_spec = b.values[beta].spec
            if tuple(g_spec.shape) != (N,) or tuple(be_spec.shape) != (N,):
                raise ValueError(f"LayerNormFwd gamma/beta must be (N,) where N={N}")
            if g_spec.dtype != x_spec.dtype or be_spec.dtype != x_spec.dtype:
                raise ValueError("LayerNormFwd gamma/beta dtype must match x dtype")
            if g_spec.device != x_spec.device or be_spec.device != x_spec.device:
                raise ValueError("LayerNormFwd gamma/beta device must match x device")
            ins = [x, gamma, beta]
        else:
            if len(rest) != 0:
                raise ValueError(f"LayerNormFwd(noaff) expects (x) only; got {1+len(rest)} args")
            ins = [x]

        emit_layernorm_fwd(
            b, ctx,
            inputs=ins,
            outputs=[y, mean, rstd],
            eps=self.eps,
            name=f"{self.name}.layernorm_fwd",
        )
        return y, mean, rstd


class LayerNormBwd(Layer):
    """
    LayerNorm backward (2D only): x,dy shape (M,N)

    affine=False:
      inputs : [x, dy, mean(fp32[M]), rstd(fp32[M])]
      outputs: [dx]

    affine=True:
      inputs : [x, dy, gamma(N), mean, rstd]
      outputs: [dx, dgamma(fp32[N]), dbeta(fp32[N])]
    """

    def __init__(self, name: str, *, affine: bool = False):
        super().__init__(name)
        self.affine = bool(affine)

    def emit(self, b, x: int, dy: int, *rest: int, ctx: CudaEmitContext):
        x_spec = b.values[x].spec
        dy_spec = b.values[dy].spec
        if len(x_spec.shape) != 2:
            raise ValueError(f"LayerNormBwd expects 2D (M,N); got shape={x_spec.shape}")
        if tuple(dy_spec.shape) != tuple(x_spec.shape):
            raise ValueError(f"LayerNormBwd shape mismatch: x={x_spec.shape} dy={dy_spec.shape}")
        if dy_spec.dtype != x_spec.dtype or dy_spec.device != x_spec.device:
            raise ValueError("LayerNormBwd dtype/device mismatch between x and dy")

        M, N = x_spec.shape
        dx = b.value(f"{self.name}.dx", TensorSpec(shape=(M, N), dtype=x_spec.dtype, device=x_spec.device))

        if self.affine:
            if len(rest) != 3:
                raise ValueError(f"LayerNormBwd(affine) expects (x,dy,gamma,mean,rstd); got {2+len(rest)} args")
            gamma, mean, rstd = rest
            g_spec = b.values[gamma].spec
            m_spec = b.values[mean].spec
            r_spec = b.values[rstd].spec

            if tuple(g_spec.shape) != (N,):
                raise ValueError(f"LayerNormBwd gamma must be (N,) where N={N}")
            if g_spec.dtype != x_spec.dtype or g_spec.device != x_spec.device:
                raise ValueError("LayerNormBwd gamma dtype/device must match x")
            if tuple(m_spec.shape) != (M,) or tuple(r_spec.shape) != (M,):
                raise ValueError(f"LayerNormBwd mean/rstd must be (M,) where M={M}")
            if m_spec.dtype != "f32" or r_spec.dtype != "f32":
                raise ValueError("LayerNormBwd mean/rstd must be f32")
            if m_spec.device != x_spec.device or r_spec.device != x_spec.device:
                raise ValueError("LayerNormBwd mean/rstd device must match x")

            dgamma = b.value(f"{self.name}.dgamma", TensorSpec(shape=(N,), dtype="f32", device=x_spec.device))
            dbeta = b.value(f"{self.name}.dbeta", TensorSpec(shape=(N,), dtype="f32", device=x_spec.device))

            emit_layernorm_bwd(
                b, ctx,
                inputs=[x, dy, gamma, mean, rstd],
                outputs=[dx, dgamma, dbeta],
                name=f"{self.name}.layernorm_bwd",
            )
            return dx, dgamma, dbeta

        else:
            if len(rest) != 2:
                raise ValueError(f"LayerNormBwd(noaff) expects (x,dy,mean,rstd); got {2+len(rest)} args")
            mean, rstd = rest
            m_spec = b.values[mean].spec
            r_spec = b.values[rstd].spec
            if tuple(m_spec.shape) != (M,) or tuple(r_spec.shape) != (M,):
                raise ValueError(f"LayerNormBwd mean/rstd must be (M,) where M={M}")
            if m_spec.dtype != "f32" or r_spec.dtype != "f32":
                raise ValueError("LayerNormBwd mean/rstd must be f32")
            if m_spec.device != x_spec.device or r_spec.device != x_spec.device:
                raise ValueError("LayerNormBwd mean/rstd device must match x")

            emit_layernorm_bwd(
                b, ctx,
                inputs=[x, dy, mean, rstd],
                outputs=[dx],
                name=f"{self.name}.layernorm_bwd",
            )
            return dx
