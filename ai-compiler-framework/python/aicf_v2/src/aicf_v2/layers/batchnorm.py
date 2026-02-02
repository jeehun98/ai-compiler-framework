from __future__ import annotations

from .base import Layer
from ..tensor_spec import TensorSpec

from ..emitters.cuda.context import CudaEmitContext
from ..emitters.cuda.batchnorm_fwd import batchnorm_fwd as emit_batchnorm_fwd
from ..emitters.cuda.batchnorm_bwd import batchnorm_bwd as emit_batchnorm_bwd


class BatchNormFwd(Layer):
    """
    BatchNorm forward.

    training (use_running_stats=False):
      inputs:  [x] or [x, gamma, beta]
      outputs: [y, save_mean(fp32[C]), save_rstd(fp32[C])]

    inference (use_running_stats=True):
      inputs:  [x, running_mean, running_var] or [x, gamma, beta, running_mean, running_var]
      outputs: [y]

    attrs(schema BNEP): eps(float32), flags(uint32) where flags bit0 = use_running_stats
    """

    def __init__(
        self,
        name: str,
        *,
        eps: float = 1e-5,
        use_running_stats: bool,
        affine: bool = True,
    ):
        super().__init__(name)
        self.eps = float(eps)
        self.use_running_stats = bool(use_running_stats)
        self.affine = bool(affine)

    def emit(self, b, x: int, *rest: int, ctx: CudaEmitContext):
        x_spec = b.values[x].spec

        # shape/rank checks: expect NCHW 4D
        if len(x_spec.shape) != 4:
            raise ValueError(f"BatchNormFwd expects 4D NCHW; got shape={x_spec.shape}")
        N, C, H, W = x_spec.shape  # noqa: F841 (N,H,W not used)

        # y spec == x spec
        y_spec = TensorSpec(shape=x_spec.shape, dtype=x_spec.dtype, device=x_spec.device)
        y = b.value(f"{self.name}.y", y_spec)

        if not self.use_running_stats:
            # -------------------------
            # training
            # -------------------------
            if self.affine:
                if len(rest) != 2:
                    raise ValueError(
                        f"BatchNormFwd(training, affine) expects (x, gamma, beta), got {1 + len(rest)} args"
                    )
                gamma, beta = rest
                ins = [x, gamma, beta]
            else:
                if len(rest) != 0:
                    raise ValueError(
                        f"BatchNormFwd(training, noaff) expects (x) only, got {1 + len(rest)} args"
                    )
                ins = [x]

            # save stats (fp32[C]) always produced in training
            stat_spec = TensorSpec(shape=(C,), dtype="f32", device=x_spec.device)
            save_mean = b.value(f"{self.name}.save_mean", stat_spec)
            save_rstd = b.value(f"{self.name}.save_rstd", stat_spec)

            emit_batchnorm_fwd(
                b, ctx,
                inputs=ins,
                outputs=[y, save_mean, save_rstd],
                eps=self.eps,
                use_running_stats=False,
                name=f"{self.name}.batchnorm_fwd",
            )
            return y, save_mean, save_rstd

        else:
            # -------------------------
            # inference
            # -------------------------
            if self.affine:
                if len(rest) != 4:
                    raise ValueError(
                        f"BatchNormFwd(infer, affine) expects (x, gamma, beta, running_mean, running_var), got {1 + len(rest)} args"
                    )
                gamma, beta, running_mean, running_var = rest
                ins = [x, gamma, beta, running_mean, running_var]
            else:
                if len(rest) != 2:
                    raise ValueError(
                        f"BatchNormFwd(infer, noaff) expects (x, running_mean, running_var), got {1 + len(rest)} args"
                    )
                running_mean, running_var = rest
                ins = [x, running_mean, running_var]

            emit_batchnorm_fwd(
                b, ctx,
                inputs=ins,
                outputs=[y],
                eps=self.eps,
                use_running_stats=True,
                name=f"{self.name}.batchnorm_fwd",
            )
            return y


class BatchNormBwd(Layer):
    """
    BatchNorm backward (training).

    inputs : x, dy, gamma, save_mean, save_rstd
    outputs: dx, dgamma(fp32[C]), dbeta(fp32[C])
    """

    def __init__(self, name: str):
        super().__init__(name)

    def emit(self, b, x: int, dy: int, gamma: int, save_mean: int, save_rstd: int, *, ctx: CudaEmitContext):
        x_spec = b.values[x].spec
        dy_spec = b.values[dy].spec

        if tuple(dy_spec.shape) != tuple(x_spec.shape):
            raise ValueError(f"BatchNormBwd shape mismatch: x={x_spec.shape} dy={dy_spec.shape}")
        if dy_spec.dtype != x_spec.dtype or dy_spec.device != x_spec.device:
            raise ValueError("BatchNormBwd dtype/device mismatch between x and dy")
        if len(x_spec.shape) != 4:
            raise ValueError(f"BatchNormBwd expects 4D NCHW; got shape={x_spec.shape}")
        _, C, _, _ = x_spec.shape

        dx = b.value(f"{self.name}.dx", TensorSpec(shape=x_spec.shape, dtype=x_spec.dtype, device=x_spec.device))
        dgamma = b.value(f"{self.name}.dgamma", TensorSpec(shape=(C,), dtype="f32", device=x_spec.device))
        dbeta = b.value(f"{self.name}.dbeta", TensorSpec(shape=(C,), dtype="f32", device=x_spec.device))

        emit_batchnorm_bwd(
            b, ctx,
            x=x, dy=dy, gamma=gamma, save_mean=save_mean, save_rstd=save_rstd,
            out_dx=dx, out_dgamma=dgamma, out_dbeta=dbeta,
            name=f"{self.name}.batchnorm_bwd",
        )
        return dx, dgamma, dbeta
