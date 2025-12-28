from __future__ import annotations
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------
# Import path fix: allow running from anywhere
# ---------------------------------------------------------------------
THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]  # .../examples/python
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))


from aicf_fw.backend.aicf_backend import AICFBackend
from aicf_fw.backend import set_backend, get_backend
from aicf_fw.core.tensor import Tensor
from aicf_fw.core.autograd import backward as autograd_backward
from aicf_fw.nn.linear import Linear
from aicf_fw.nn.relu import ReLU
from aicf_fw.nn.sequential import Sequential
from aicf_fw.optim.sgd import SGD


@torch.no_grad()
def snapshot_params(model: Sequential):
    snaps = {}
    for n, p in model.named_parameters():
        snaps[n] = p.data.detach().clone()
    return snaps


def max_param_diff(model: Sequential, snaps) -> float:
    m = 0.0
    for n, p in model.named_parameters():
        d = (p.data - snaps[n]).abs().max().item()
        if d > m:
            m = float(d)
    return m


def main():
    assert torch.cuda.is_available()

    backend = AICFBackend()
    set_backend(backend)
    bk = get_backend()

    # 매우 중요: 시작부터 캡처 상태 초기화
    backend.capture_reset()

    model = Sequential(
        Linear(8, 8, device="cuda", dtype=torch.float32),
        ReLU(),
        Linear(8, 8, device="cuda", dtype=torch.float32),
    )

    optim = SGD(model, lr=1e-4, inplace=True, grad_clip=5.0)

    # 고정 입력 (shape 고정)
    x = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False)
    t = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False)

    for n, p in model.named_parameters():
        print("[param]", n, tuple(p.data.shape), p.data.dtype, p.data.device)

    # ----------------------------
    # AICF-only train step
    # - mse_loss forward(mean 등) 제거
    # - mse_grad만 쓰고, y에 대한 seed grad로 역전파
    # ----------------------------
    def train_step_aicf_only():
        y = model(x)  # forward (AICF ops)
        optim.zero_grad()

        # dY = mse_grad(y, t)  (AICF op)
        dY = bk.op_call("mse_grad", [y.data, t.data], {})
        dY_t = Tensor(dY, requires_grad=False)

        # seed gradient: backward(y, grad=dY)
        autograd_backward(y, grad=dY_t)

        # step (AICF sgd_step in-place)
        optim.step()

        # 디버그용 스칼라(loss 비슷한 값): torch에서만 계산(캡처 밖에서만 쓸 것)
        # 캡처 구간에서 절대 쓰지 말 것.
        with torch.no_grad():
            diff = (y.data - t.data)
            loss_scalar = float((diff * diff).mean().detach().cpu().item())
        return loss_scalar

    # ----------------------------
    # Warmup (캡처 밖)
    # ----------------------------
    loss0 = train_step_aicf_only()
    torch.cuda.synchronize()
    print("[warmup] loss_like =", loss0)

    # warmup 후에도 혹시 남아있을 수 있는 상태 제거
    backend.capture_reset()
    torch.cuda.synchronize()

    # ----------------------------
    # Capture 1 iteration
    # ----------------------------
    backend.capture_begin()
    # 캡처 안에서는 torch 연산/print/clone 금지
    _ = model(x)
    optim.zero_grad()
    dY = bk.op_call("mse_grad", [_.data, t.data], {})
    autograd_backward(_, grad=Tensor(dY, requires_grad=False))
    optim.step()
    backend.capture_end()

    torch.cuda.synchronize()
    print("[capture] done")

    # ----------------------------
    # Replay loop
    # ----------------------------
    snaps = snapshot_params(model)

    reps = 20
    for i in range(reps):
        backend.replay()
        torch.cuda.synchronize()

        # replay 이후에만 torch로 loss_like 확인
        with torch.no_grad():
            y = model(x)  # NOTE: 이건 캡처 밖 실행(확인용). 원하면 제거 가능.
            diff = (y.data - t.data)
            loss_like = float((diff * diff).mean().detach().cpu().item())

        diffp = max_param_diff(model, snaps)
        snaps = snapshot_params(model)

        print(f"[replay {i:02d}] loss_like={loss_like:.10f} param_step_maxdiff={diffp:.6e}")

    print("OK")


if __name__ == "__main__":
    main()
