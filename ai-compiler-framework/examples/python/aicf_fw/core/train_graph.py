from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Literal

import torch

from aicf_fw.core.tensor import Tensor
from aicf_fw.core.autograd import backward as autograd_backward
from aicf_fw.core import functional as F
from aicf_fw.core.compile import compile_and_capture
from aicf_fw.core.artifact import CompileArtifact


LossKind = Literal["mse"]
# spec: name -> (shape, dtype, device)
InputSpec = Dict[str, Tuple[Tuple[int, ...], torch.dtype, str]]


def _make_static_tensor(shape: Tuple[int, ...], dtype: torch.dtype, device: str, name: str) -> Tensor:
    t = torch.zeros(shape, device=device, dtype=dtype)
    return Tensor(t, requires_grad=False, name=name)


class TrainGraph:
    """
    Owns static input buffers and a captured CUDA graph for one training step.
    User updates inputs via set_inputs(...), then calls replay().
    """

    def __init__(
        self,
        model: Any,
        optim: Any,
        *,
        input_spec: InputSpec,
        loss: LossKind = "mse",
        name: str = "train_step",
        warmup_runs: int = 2,
        warmup_sync: bool = True,
        validate: bool = True,
        trace: bool = True,
        enforce_ops: Tuple[str, ...] = ("adam_step",),
        torch_sync: bool = True,
    ):
        if loss != "mse":
            raise ValueError("v0 supports only loss='mse'")

        self.model = model
        self.optim = optim
        self.loss = loss
        self.name = name

        # create static buffers
        self.inputs: Dict[str, Tensor] = {}
        for k, (shape, dtype, device) in input_spec.items():
            self.inputs[k] = _make_static_tensor(shape, dtype, device, name=k)

        # Build step_fn that always uses static buffers
        def _step_fn():
            x = self.inputs["x"]
            t = self.inputs["t"]

            self.optim.zero_grad()
            y = self.model(x)
            dY = F.mse_grad(y, t)
            autograd_backward(y, grad=dY, accumulate=False)
            self.optim.step_()

        self._step_fn = _step_fn

        # Compile+Capture
        self.artifact: CompileArtifact = compile_and_capture(
            self._step_fn,
            name=name,
            warmup_runs=warmup_runs,
            warmup_sync=warmup_sync,
            validate=validate,
            trace=trace,
            enforce_ops=enforce_ops,
            torch_sync=torch_sync,
        )

    def set_inputs(self, **kwargs: torch.Tensor) -> None:
        """
        Copy external torch tensors into static input buffers.
        Usage: set_inputs(x=torch_tensor, t=torch_tensor)
        """
        for k, v in kwargs.items():
            if k not in self.inputs:
                raise KeyError(f"Unknown input '{k}'. Known: {list(self.inputs.keys())}")
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"set_inputs expects torch.Tensor for '{k}', got {type(v)}")

            dst = self.inputs[k].data
            if tuple(v.shape) != tuple(dst.shape):
                raise ValueError(f"shape mismatch for input '{k}': {tuple(v.shape)} vs {tuple(dst.shape)}")
            if v.dtype != dst.dtype:
                v = v.to(dtype=dst.dtype)
            if v.device != dst.device:
                v = v.to(device=dst.device)

            dst.copy_(v)

    def replay(self) -> None:
        self.artifact.backend.replay()

    # ---- convenience passthroughs ----
    def dump(self, *, ir: bool = True, lowered: bool = True, trace: bool = True) -> None:
        if ir:
            print("=== IR DUMP ===")
            print(self.artifact.ir.dump_json(indent=2))
        if lowered:
            print("=== LOWERED OPS ===")
            for i, it in enumerate(self.artifact.lowered):
                print(f"[lower {i:02d}] op={it['op']} attrs={it['attrs']}")
        if trace:
            print("=== TRACE OPS (runtime) ===")
            for i, op in enumerate(self.artifact.trace_ops):
                print(f"[trace {i:02d}] op={op}")

    def assert_runtime_matches_lowering(self, *, trace_filter: bool = True) -> None:
        self.artifact.assert_runtime_matches_lowering(self.model, trace_filter=trace_filter)

    def assert_adam_state_mutates(self, *, tag: str = "smoke") -> None:
        self.artifact.assert_adam_state_mutates(self.model, self.optim, tag=tag)

    def assert_determinism(self, *, replays: int = 20, check_restore: bool = True) -> None:
        self.artifact.assert_determinism(self.model, self.optim, replays=replays, check_restore=check_restore)
