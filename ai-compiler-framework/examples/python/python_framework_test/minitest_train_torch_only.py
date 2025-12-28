from __future__ import annotations
import torch


def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available()

    device = "cuda"
    dtype = torch.float32

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8, bias=True, device=device, dtype=dtype),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 8, bias=True, device=device, dtype=dtype),
    )

    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    x = torch.randn(64, 8, device=device, dtype=dtype)
    t = torch.randn(64, 8, device=device, dtype=dtype)

    def step():
        opt.zero_grad(set_to_none=True)
        y = model(x)
        loss = ((y - t) ** 2).mean()
        loss.backward()
        opt.step()
        return float(loss.detach().cpu().item())

    # warmup
    l0 = step()
    torch.cuda.synchronize()
    print("[torch-only warmup] loss =", l0)

    # run
    for i in range(20):
        l = step()
        torch.cuda.synchronize()
        print(f"[torch-only {i:02d}] loss={l:.10f}")

    print("OK")


if __name__ == "__main__":
    main()
