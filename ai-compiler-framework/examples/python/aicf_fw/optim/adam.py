import torch
import aicf_cuda as aicf

class Adam:
    """
    AICF Adam (f32) using fused OpKind.AdamStep.

    - params: iterable of aicf_fw.core.tensor.Tensor (Parameter wrapper)
    - state m/v: torch.Tensor (same shape/dtype/device as param.data)
    - step uses: aicf.op_call(OpKind.AdamStep, [p.data, p.grad.data, m, v] -> [p.data, m, v], attrs)
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)  # list of AICF Tensor wrappers
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

        self.t = 0
        self.m = [None] * len(self.params)
        self.v = [None] * len(self.params)

    def warmup_state(self):
        """
        Must be called OUTSIDE capture:
        - allocate m/v buffers
        - zero-initialize
        """
        for i, p in enumerate(self.params):
            td = p.data  # torch.Tensor
            if self.m[i] is None:
                self.m[i] = torch.empty_like(td)
                self.v[i] = torch.empty_like(td)
                # capture-safe init not required here because we're outside capture anyway,
                # but we can still use GradZero op to keep policy consistent.
                aicf.op_call(aicf.OpKind.GradZero, [self.m[i]], [self.m[i]], {})
                aicf.op_call(aicf.OpKind.GradZero, [self.v[i]], [self.v[i]], {})

    def step(self):
        """
        Graph-safe step: NO allocations, only op_call.
        Assumes warmup_state() already materialized m/v.
        """
        # Ensure state exists (safety net; avoid KeyError/None)
        # IMPORTANT: this does not allocate if warmup_state was called.
        for i, p in enumerate(self.params):
            if self.m[i] is None or self.v[i] is None:
                raise RuntimeError("Adam.step called before warmup_state(): m/v not materialized")

        self.t += 1
        bc1_inv = 1.0 / (1.0 - (self.beta1 ** self.t))
        bc2_inv = 1.0 / (1.0 - (self.beta2 ** self.t))

        attrs = {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "bc1_inv": float(bc1_inv),
            "bc2_inv": float(bc2_inv),
        }

        for i, p in enumerate(self.params):
            # p: AICF Tensor wrapper
            if p.grad is None:
                continue  # nothing to update

            # grad is also AICF Tensor wrapper
            g = p.grad.data
            aicf.op_call(
                aicf.OpKind.AdamStep,
                [p.data, g, self.m[i], self.v[i]],
                [p.data, self.m[i], self.v[i]],
                attrs,
            )
