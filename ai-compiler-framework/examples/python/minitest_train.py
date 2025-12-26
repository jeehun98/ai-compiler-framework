# examples/python/minitest_train.py
import torch
from aicf_fw.backend import set_backend
from aicf_fw.backend.torch_backend import TorchBackend
from aicf_fw.core.tensor import Tensor
from aicf_fw.nn.linear import Linear
from aicf_fw.nn.relu import ReLU
from aicf_fw.nn.sequential import Sequential
from aicf_fw.nn.losses import MSELoss
from aicf_fw.nn.optim.sgd import SGD

def main():
    set_backend(TorchBackend())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    model = Sequential(
        Linear(16, 32, dtype=dtype, device=device),
        ReLU(),
        Linear(32, 8, dtype=dtype, device=device),
    )

    loss_fn = MSELoss()
    opt = SGD(model.parameters(), lr=1e-3)

    x = Tensor(torch.randn(64, 16, device=device, dtype=dtype), requires_grad=False)
    t = Tensor(torch.randn(64, 8, device=device, dtype=dtype), requires_grad=False)

    for it in range(20):
        opt.zero_grad()
        y = model(x)
        loss = loss_fn(y, t)
        loss.backward()
        opt.step()
        print(it, float(loss.data))

if __name__ == "__main__":
    main()
