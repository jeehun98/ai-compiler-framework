from __future__ import annotations

import torch

from aicf_fw.core.autograd import Tensor, TensorMeta
from aicf_fw.nn.sequential import Sequential
from aicf_fw.core import functional as F
from aicf_fw.core.compile import is_tracing
from aicf_fw.core.autograd import in_capture


class Adam:
    """
    Capture-safe Adam (stateful) for AICF FW.

    Key policy:
      - During capture, every param MUST have a materialized (pointer-stable) grad buffer.
      - If p.grad is None during capture, that's a bug (warmup missing / someone set grad=None).
        We fail fast instead of silently skipping (which produces "no-op" training graphs).
    """

    def __init__(
        self,
        model: Sequential,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        grad_clip=None,
    ):
        self.model = model
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.grad_clip = grad_clip

        self.params = [p for _, p in model.named_parameters()]

        self.m = {}   # param_index -> Tensor
        self.v = {}   # param_index -> Tensor

        self.step = None
        self.bc1_inv = None
        self.bc2_inv = None

        self._init_state()

    def _init_state(self):
        assert len(self.params) > 0
        dev = self.params[0].data.device

        self.step = torch.zeros((), device=dev, dtype=torch.int32)
        self.bc1_inv = torch.empty((), device=dev, dtype=torch.float32)
        self.bc2_inv = torch.empty((), device=dev, dtype=torch.float32)

        for i, p in enumerate(self.params):
            self.m[i] = Tensor(torch.zeros_like(p.data), requires_grad=False)
            self.v[i] = Tensor(torch.zeros_like(p.data), requires_grad=False)

    def zero_grad(self):
        # capture-safe grad reset (does NOT set grad=None)
        for p in self.params:
            if p.grad is None:
                continue
            F.grad_zero_(p.grad)

    @torch.no_grad()
    def step_update(self):
        self.step_()

    @torch.no_grad()
    def step_(self):
        """
        One optimizer step (capture-safe).

        TRACING:
          - emit StepInc/BiasCorr/AdamStep nodes even if grads are symbolic.
        RUNTIME:
          - during capture, grad must exist (fail fast if None).
        """
        # --------------------------------------------------------
        # TRACING PATH
        # --------------------------------------------------------
        if is_tracing():
            F.step_inc_(self.step)
            F.bias_corr_out(self.step, self.bc1_inv, self.bc2_inv, self.beta1, self.beta2)

            for i, p in enumerate(self.params):
                gmeta = TensorMeta(shape=p.shape, dtype=p.dtype, device=p.device)
                gsym = Tensor(None, requires_grad=False, name=(p.name + ".grad") if p.name else "grad", meta=gmeta)

                F.adam_step_(
                    p=p,
                    g=gsym,
                    m=self.m[i],
                    v=self.v[i],
                    bc1_inv=self.bc1_inv,
                    bc2_inv=self.bc2_inv,
                    lr=self.lr,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    eps=self.eps,
                )
            return

        # --------------------------------------------------------
        # EXECUTION PATH
        # --------------------------------------------------------
        F.step_inc_(self.step)
        F.bias_corr_out(self.step, self.bc1_inv, self.bc2_inv, self.beta1, self.beta2)

        cap = in_capture()

        for i, p in enumerate(self.params):
            g = p.grad

            # ★ 핵심 수정: capture 중 grad=None이면 "조용히 continue" 금지
            if g is None:
                if cap:
                    raise RuntimeError(
                        "Adam.step_: p.grad is None during capture. "
                        "Warmup must materialize ALL parameter.grad buffers and you must not set them to None. "
                        f"(param_index={i}, param_name={getattr(p, 'name', '')})"
                    )
                continue

            # optional grad clip (runtime-only)
            if (self.grad_clip is not None) and (not cap):
                gn = g.data.norm().item()
                if gn > self.grad_clip:
                    g.data.mul_(self.grad_clip / (gn + 1e-12))

            F.adam_step_(
                p=p,
                g=g,
                m=self.m[i],
                v=self.v[i],
                bc1_inv=self.bc1_inv,
                bc2_inv=self.bc2_inv,
                lr=self.lr,
                beta1=self.beta1,
                beta2=self.beta2,
                eps=self.eps,
            )
