from __future__ import annotations

import torch

from aicf_fw.core.tensor import Tensor, TensorMeta
from aicf_fw.nn.sequential import Sequential
from aicf_fw.core import functional as F
from aicf_fw.core.trace import is_tracing


class Adam:
    """
    Capture-safe Adam (stateful) for AICF FW.
    - params: Sequential model
    - maintains m,v per param tensor (f32)
    - maintains step (int32 scalar on CUDA)
    - bias correction computed by CUDA ops: StepInc + BiasCorr
    - update via AdamStep CUDA op (in-place)
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

        # grab params (Tensor objects)
        self.params = [p for _, p in model.named_parameters()]

        # --- state buffers (materialize OUTSIDE capture via warmup) ---
        self.m = {}   # param_index -> Tensor
        self.v = {}   # param_index -> Tensor

        # step + biascorr scalars live as torch tensors (so we can feed to op_call)
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
        # capture-safe grad reset
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

        TRACING policy:
          - During IR compile/tracing, p.grad may not exist (symbolic).
          - We still want to emit StepInc/BiasCorr/AdamStep nodes.
          - So we create a symbolic grad tensor with same meta as p and emit AdamStep.
        """
        # --------------------------------------------------------
        # TRACING PATH (compile/IR)
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
        # EXECUTION PATH (existing)
        # --------------------------------------------------------
        F.step_inc_(self.step)
        F.bias_corr_out(self.step, self.bc1_inv, self.bc2_inv, self.beta1, self.beta2)

        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue

            # optional grad clip (runtime-only)
            if self.grad_clip is not None:
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
