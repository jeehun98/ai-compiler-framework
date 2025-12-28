from __future__ import annotations

from typing import Callable, Optional
import torch

from aicf_fw.core.tensor import Tensor
from aicf_fw.backend import get_backend


@torch.no_grad()
def warmup_capture_safe(
    *,
    train_step: Callable[[], None],
    runs: int = 1,
    sync: bool = True,
) -> None:
    """
    Materialize all capture-required buffers BEFORE capture.

    Guarantees (when combined with enforced capture rules):
      - Functional BufferPool buffers are allocated (stable pointers).
      - Leaf parameter.grad buffers are allocated (stable pointers).
      - Any lazy initialization in kernels/dispatch is forced to happen outside capture.

    Usage:
      warmup_capture_safe(train_step=train_step_aicf_only, runs=1)
      backend.capture_begin(); ...; backend.capture_end()
    """
    if runs <= 0:
        return

    bk = get_backend()

    # Make sure we are NOT in capture
    # (If you accidentally call this inside capture, it will break the whole policy.)
    from aicf_fw.core.autograd import in_capture
    if in_capture():
        raise RuntimeError("warmup_capture_safe() must be called OUTSIDE capture.")

    # Run train_step a few times to materialize:
    # - BufferPool activations/grads
    # - leaf.grad buffers
    for _ in range(runs):
        train_step()

    if sync:
        torch.cuda.synchronize()
