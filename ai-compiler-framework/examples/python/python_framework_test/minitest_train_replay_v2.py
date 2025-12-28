from __future__ import annotations
import sys
from pathlib import Path
import torch

THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

from aicf_fw.backend.aicf_backend import AICFBackend  # NOTE: if your actual path is aicf_fw.backend.aicf_backend import AICFBackend, use that
from aicf_fw.backend import set_backend, get_backend
from aicf_fw.core.tensor import Tensor
from aicf_fw.core.autograd import backward as autograd_backward
from aicf_fw.nn.linear import Linear
from aicf_fw.nn.relu import ReLU
from aicf_fw.nn.sequential import Sequential
from aicf_fw.optim.sgd import SGD


@torch.no_grad()
def snapshot_params(model: Sequential):
    return {n: p.data.detach().clone() for n, p in model.named_parameters()}


def max_param_diff(model: Sequential, snaps) -> float:
    m = 0.0
    for n, p in model.named_parameters():
        d = (p.data - snaps[n]).abs().max().item()
        if d > m:
            m = float(d)
    return m


@torch.no_grad()
def loss_like(model: Sequential, x: Tensor, t: Tensor) -> float:
    y = model(x)
    diff = (y.data - t.data)
    return float((diff * diff).mean().detach().cpu().item())


def main():
    assert torch.cuda.is_available()

    backend = AICFBackend()
    set_backend(backend)
    bk = get_backend()

    backend.capture_reset()
    torch.cuda.synchronize()

    model = Sequential(
        Linear(8, 8, device="cuda", dtype=torch.float32),
        ReLU(),
        Linear(8, 8, device="cuda", dtype=torch.float32),
    )
    optim = SGD(model, lr=1e-4, inplace=True, grad_clip=None)

    x = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False)
    t = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False)

    for n, p in model.named_parameters():
        print("[param]", n, tuple(p.data.shape), p.data.dtype, p.data.device)

    # ----------------------------
    # Warmup 1: materialize leaf grad buffers
    # ----------------------------
    def train_step_aicf_only(dY_buf: torch.Tensor):
        y = model(x)

        # IMPORTANT: do NOT allocate dY during capture.
        # Write into preallocated dY_buf.
        bk.op_call_out("mse_grad", [y.data, t.data], [dY_buf], {})

        autograd_backward(y, grad=Tensor(dY_buf, requires_grad=False), accumulate=False)
        optim.step()

    # warmup: run once to create parameter.grad buffers
    # also uses a temporary dY buffer for this warmup
    with torch.no_grad():
        y_tmp = model(x)  # outside capture
    dY_buf = torch.empty_like(y_tmp.data)  # persistent buffer (pointer stable)

    train_step_aicf_only(dY_buf)
    torch.cuda.synchronize()
    print("[warmup] loss_like =", loss_like(model, x, t))

    # ----------------------------
    # Capture
    # ----------------------------
    backend.capture_reset()
    torch.cuda.synchronize()

    backend.capture_begin()
    train_step_aicf_only(dY_buf)
    backend.capture_end()

    torch.cuda.synchronize()
    print("[capture] done")

    # ----------------------------
    # Replay loop
    # ----------------------------
    snaps = snapshot_params(model)
    for i in range(20):
        backend.replay()
        torch.cuda.synchronize()

        l = loss_like(model, x, t)
        diffp = max_param_diff(model, snaps)
        snaps = snapshot_params(model)
        print(f"[replay {i:02d}] loss_like={l:.10f} param_step_maxdiff={diffp:.6e}")

    print("OK")


if __name__ == "__main__":
    main()
