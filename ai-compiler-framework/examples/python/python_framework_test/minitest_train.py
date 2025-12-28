# examples/python/minitest_train.py
from __future__ import annotations

import torch

from aicf_fw.backend.aicf_backend import AICFBackend
from aicf_fw.backend import set_backend
from aicf_fw.core.tensor import Tensor
from aicf_fw.nn.linear import Linear
from aicf_fw.nn.relu import ReLU
from aicf_fw.nn.sequential import Sequential
from aicf_fw.nn.losses import MSELoss
from aicf_fw.optim.sgd import SGD


def collect_params(model) -> list[Tensor]:
    """
    aicf_fw에 parameters()가 없을 수 있어서,
    known module types를 기준으로 직접 파라미터 수집.
    """
    params: list[Tensor] = []

    if hasattr(model, "layers"):
        for layer in model.layers:
            params += collect_params(layer)
        return params

    if hasattr(model, "W"):
        params.append(model.W)
    if hasattr(model, "b") and model.b is not None:
        params.append(model.b)

    return params


def main():
    assert torch.cuda.is_available()

    # backend 세팅
    backend = AICFBackend()
    set_backend(backend)

    # 모델
    model = Sequential(
        Linear(8, 8, device="cuda"),
        ReLU(),
        Linear(8, 8, device="cuda"),
    )

    loss_fn = MSELoss()

    # 파라미터 수집 + 옵티마이저 (루프 밖에서 1회 생성)
    params = collect_params(model)
    lr = 1e-4
    optim = SGD(params, lr=lr, inplace=True, grad_clip=5.0)

    # dummy data
    x = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False)
    t = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False)

    for step in range(20):
        # forward
        y = model(x)
        loss = loss_fn(y, t)

        # backward
        optim.zero_grad()
        loss.backward()

        # debug: weight/grad finite + update magnitude
        W = model.layers[0].W.data
        gW = model.layers[0].W.grad.data

        finite_W = torch.isfinite(W).all().item()
        finite_gW = torch.isfinite(gW).all().item()

        Wmax = float(W.abs().max().item())
        gmax = float(gW.abs().max().item())

        # measure update size
        W_before = W.clone()
        optim.step()
        upd_max = float((model.layers[0].W.data - W_before).abs().max().item())

        print(
            "finite?", finite_W, finite_gW,
            "Wmax", Wmax, "gmax", gmax,
            "upd_max", upd_max,
        )
        print(step, f"{float(loss.data.detach().cpu().item()):.10f}")

    # optional: final sync
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
