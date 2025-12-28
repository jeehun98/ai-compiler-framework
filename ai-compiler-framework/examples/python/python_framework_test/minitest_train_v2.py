# examples/python/minitest_train_v2.py
from __future__ import annotations
# examples/python/python_framework_test/test_gemm_unified_strided_cli.py

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

import torch

from aicf_fw.backend.aicf_backend import AICFBackend
from aicf_fw.backend import set_backend
from aicf_fw.core.tensor import Tensor
from aicf_fw.nn.linear import Linear
from aicf_fw.nn.relu import ReLU
from aicf_fw.nn.sequential import Sequential
from aicf_fw.nn.losses import MSELoss
from aicf_fw.optim.sgd import SGD


def main():
    assert torch.cuda.is_available()

    # backend 세팅
    backend = AICFBackend()
    set_backend(backend)

    # 모델
    model = Sequential(
        Linear(8, 8, device="cuda", dtype=torch.float32),
        ReLU(),
        Linear(8, 8, device="cuda", dtype=torch.float32),
    )

    loss_fn = MSELoss()

    # [UPDATED] collect_params 제거: Optimizer가 model에서 자동 수집
    lr = 1e-4
    optim = SGD(model, lr=lr, inplace=True, grad_clip=5.0)

    # dummy data
    x = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False)
    t = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False)

    # optional: print param names once
    for n, p in model.named_parameters():
        print("[param]", n, tuple(p.data.shape), p.data.dtype, p.data.device)

    for step in range(20):
        # forward
        y = model(x)
        loss = loss_fn(y, t)

        # backward
        optim.zero_grad()
        loss.backward()

        # debug: finite + update magnitude
        W0 = model.layers[0].W.data
        gW0 = model.layers[0].W.grad.data

        finite_W = torch.isfinite(W0).all().item()
        finite_gW = torch.isfinite(gW0).all().item()

        Wmax = float(W0.abs().max().item())
        gmax = float(gW0.abs().max().item())

        W_before = W0.clone()
        optim.step()
        upd_max = float((model.layers[0].W.data - W_before).abs().max().item())

        print(
            "finite?", finite_W, finite_gW,
            "Wmax", Wmax, "gmax", gmax,
            "upd_max", upd_max,
            "loss", float(loss.data.detach().cpu().item()),
        )

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
