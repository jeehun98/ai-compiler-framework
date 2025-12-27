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
    parameters()가 없을 수 있어서 known module 구조를 기준으로 직접 수집.
    - Sequential: layers 순회
    - Linear: W, b 수집
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

    # 파라미터 수집 + 옵티마이저 (✅ 반드시 inplace=True)
    params = collect_params(model)
    optim = SGD(params, lr=1e-4, inplace=True, grad_clip=5.0)

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

        # debug (backward 직후)
        W0 = model.layers[0].W.data
        gW0 = model.layers[0].W.grad.data if model.layers[0].W.grad is not None else None

        if gW0 is None:
            print("finite? W:", torch.isfinite(W0).all().item(), "gW: None",
                  "Wmax", float(W0.abs().max().item()))
        else:
            print(
                "finite?",
                torch.isfinite(W0).all().item(),
                torch.isfinite(gW0).all().item(),
                "Wmax", float(W0.abs().max().item()),
                "gmax", float(gW0.abs().max().item()),
            )

        # update
        optim.step()

        # print
        print(step, float(loss.data.detach().cpu().item()))

    # (선택) CUDA sync로 마지막 커널 완료 보장
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
