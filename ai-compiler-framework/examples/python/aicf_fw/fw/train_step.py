from __future__ import annotations
import torch
from aicf_fw.core_v2.exec import PlannedExecutor, ExecOptions

class CompiledTrainStep:
    def __init__(self, *, ir, lowered, plan, ex: PlannedExecutor, params: dict[str, torch.Tensor], statics: dict[str, torch.Tensor], optimizer):
        self.ir = ir
        self.lowered = lowered
        self.plan = plan
        self.ex = ex
        self.params = params
        self.statics = statics
        self.optimizer = optimizer

    def _bind_all(self) -> dict[str, torch.Tensor]:
        # params + statics are passed as `params=` to executor in your current API
        d = {}
        d.update(self.params)
        d.update(self.statics)
        return d

    def run(self, inputs: dict[str, torch.Tensor], reuse_static: bool = True):
        # step meta updated outside if you want
        return self.ex.run(inputs=inputs, params=self._bind_all(), reuse_static=reuse_static)

    def train_step(self, inputs: dict[str, torch.Tensor], reuse_static: bool = True):
        # host meta update
        self.optimizer.update_meta()
        # safest sync (later: replace with stream/event policy)
        torch.cuda.synchronize()
        return self.run(inputs, reuse_static=reuse_static)

    def capture(self, inputs: dict[str, torch.Tensor], reuse_static: bool = True):
        if hasattr(self.ex, "reset_graph"):
            self.ex.reset_graph()
        self.ex.capture(inputs=inputs, params=self._bind_all(), reuse_static=reuse_static)

    def replay(self, n: int = 1):
        self.ex.replay(n=n)

    def reset(self):
        if hasattr(self.ex, "reset_graph"):
            self.ex.reset_graph()
