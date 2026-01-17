from __future__ import annotations
import os
import torch
from aicf_fw.core_v2.exec import PlannedExecutor

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    if v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

class CompiledTrainStep:
    def __init__(
        self,
        *,
        ir,
        lowered,
        plan,
        ex: PlannedExecutor,
        params: dict[str, torch.Tensor],
        statics: dict[str, torch.Tensor],
        optimizer,
        warmup_runs: int = 2,
        warmup_required: bool = True,
    ):
        self.ir = ir
        self.lowered = lowered
        self.plan = plan
        self.ex = ex
        self.params = params
        self.statics = statics
        self.optimizer = optimizer

        self.warmup_runs = int(warmup_runs)
        self.warmup_required = bool(warmup_required)
        self._warmed_up = False

    def _bind_all(self) -> dict[str, torch.Tensor]:
        d = {}
        d.update(self.params)
        d.update(self.statics)
        return d

    def _maybe_warmup(self, inputs: dict[str, torch.Tensor] | None):
        if self._warmed_up:
            return
        if self.warmup_runs <= 0:
            # warmup disabled: if required, fail-fast (prevents silent no-op)
            if self.warmup_required:
                raise RuntimeError(
                    "Warmup is disabled (warmup_runs=0 or AICF_WARMUP=0) but warmup_required=True.\n"
                    "This can lead to no-op updates due to unmaterialized static/state buffers.\n"
                    "Fix: set warmup_runs>=1, or pass warmup_inputs to compile_train_step(), or disable warmup_required explicitly."
                )
            return

        if inputs is None:
            raise RuntimeError(
                "Warmup is required but no inputs were provided.\n"
                "Fix: call compiled.warmup(example_inputs) once, or pass warmup_inputs=... to compile_train_step()."
            )

        self.warmup(inputs, n=self.warmup_runs)

    def warmup(self, inputs: dict[str, torch.Tensor], n: int = 2, reuse_static: bool = True):
        # warmup does NOT need meta update each iteration; but it's fine to keep deterministic
        for _ in range(int(n)):
            self.optimizer.update_meta()
            torch.cuda.synchronize()
            self.ex.run(inputs=inputs, params=self._bind_all(), reuse_static=reuse_static)
        self._warmed_up = True

    def run(self, inputs: dict[str, torch.Tensor], reuse_static: bool = True):
        return self.ex.run(inputs=inputs, params=self._bind_all(), reuse_static=reuse_static)

    def train_step(self, inputs: dict[str, torch.Tensor], reuse_static: bool = True):
        self._maybe_warmup(inputs)

        self.optimizer.update_meta()
        torch.cuda.synchronize()
        return self.run(inputs, reuse_static=reuse_static)

    def capture(self, inputs: dict[str, torch.Tensor], reuse_static: bool = True):
        # ensure warmed up before capture for pointer/alloc stability
        self._maybe_warmup(inputs)

        if hasattr(self.ex, "reset_graph"):
            self.ex.reset_graph()
        self.ex.capture(inputs=inputs, params=self._bind_all(), reuse_static=reuse_static)

    def replay(self, n: int = 1):
        self.ex.replay(n=int(n))

    def reset(self):
        if hasattr(self.ex, "reset_graph"):
            self.ex.reset_graph()
