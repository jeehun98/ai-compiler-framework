# examples/python/aicf_fw/optim/adam.py

from __future__ import annotations

import torch
from aicf_fw.core.tensor import Tensor
from aicf_fw.nn.sequential import Sequential
from aicf_fw.core import functional as F

class Adam:
    """
    Capture-safe Adam (stateful) for AICF FW.
    - params: Sequential model
    - maintains m,v per param tensor (f32)
    - maintains step (int32 scalar on CUDA)
    - bias correction computed by CUDA ops: StepInc + BiasCorr
    - update via AdamStep CUDA op (in-place)
    """

    def __init__(self,
                 model: Sequential,
                 lr: float = 1e-3,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8,
                 grad_clip=None):
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
        # step: int32 scalar CUDA
        self.step = None
        self.bc1_inv = None
        self.bc2_inv = None

        self._init_state()

    def _init_state(self):
        # allocate state on same device/dtype as params (we assume f32 params for now)
        # (you can extend dtype later)
        assert len(self.params) > 0
        dev = self.params[0].data.device

        self.step = torch.zeros((), device=dev, dtype=torch.int32)
        self.bc1_inv = torch.empty((), device=dev, dtype=torch.float32)
        self.bc2_inv = torch.empty((), device=dev, dtype=torch.float32)

        for i, p in enumerate(self.params):
            # m,v same shape, float32, requires_grad False
            # use torch.zeros_like on the raw torch tensor
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
        """
        Alias of step(); name to avoid confusion with scalar step tensor.
        """
        self.step_()

    @torch.no_grad()
    def step_(self):
        """
        One optimizer step (capture-safe).
        Requires:
          - p.grad exists and is contiguous CUDA tensor
          - autograd_backward(accumulate=False) used in capture region
        """
        # step += 1 (int32 scalar)
        F.step_inc_(self.step)

        # bias correction into bc tensors
        F.bias_corr_out(self.step, self.bc1_inv, self.bc2_inv, self.beta1, self.beta2)

        # update each parameter in-place
        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue

            # optional grad clip (keep it off during capture unless op exists)
            if self.grad_clip is not None:
                # NOT capture-safe unless you have clip op.
                # leave as runtime-only or implement clip kernel later.
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
