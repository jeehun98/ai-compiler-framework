# examples/python/aicf_fw/losses/base.py
from __future__ import annotations

class Loss:
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
