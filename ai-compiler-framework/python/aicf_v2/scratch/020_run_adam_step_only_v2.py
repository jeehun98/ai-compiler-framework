from __future__ import annotations
import sys, os
from pathlib import Path
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]  # python/aicf_v2 기준이면 상황에 맞게 조정
SRC = ROOT / "src"
sp = str(SRC)
if sp not in sys.path:
    sys.path.insert(0, sp)

import aicf_v2 as aicf


def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def adam_ref(P, G, M, V, bc1, bc2, lr, beta1, beta2, eps):
    m_new = beta1 * M + (1.0 - beta1) * G
    v_new = beta2 * V + (1.0 - beta2) * (G * G)
    m_hat = m_new / bc1
    v_hat = v_new / bc2
    P_new = P - lr * (m_hat / (torch.sqrt(v_hat) + eps))
    return P_new, m_new, v_new


def main():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float32

    lr, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8
    shape = (64, 256)

    # --- build graph ---
    m = aicf.Model(dtype="f32", device=device)

    P = m.input("P", aicf.TensorSpec(shape=shape))
    G = m.input("G", aicf.TensorSpec(shape=shape))
    M = m.input("M", aicf.TensorSpec(shape=shape))
    V = m.input("V", aicf.TensorSpec(shape=shape))
    bc1 = m.input("bc1", aicf.TensorSpec(shape=()))  # scalar
    bc2 = m.input("bc2", aicf.TensorSpec(shape=()))  # scalar

    P2, M2, V2 = m.add(aicf.AdamStep(name="adam", lr=lr, beta1=beta1, beta2=beta2, eps=eps),
                       P, G, M, V, bc1, bc2)

    m.output("P2", P2)
    m.output("M2", M2)
    m.output("V2", V2)

    print(m.dump())

    # --- feed ---
    Pt = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    Gt = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    Mt = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    Vt = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    bc1t = torch.tensor(1.0 - beta1**3, device=device, dtype=dtype).contiguous()
    bc2t = torch.tensor(1.0 - beta2**3, device=device, dtype=dtype).contiguous()

    Pref, Mref, Vref = adam_ref(Pt, Gt, Mt, Vt, bc1t, bc2t, lr, beta1, beta2, eps)

    exe = aicf.CudaExecutor()
    out = exe.run(m, {"P": Pt, "G": Gt, "M": Mt, "V": Vt, "bc1": bc1t, "bc2": bc2t})

    dP = maxabs_delta(out["P2"], Pref)  # <- output value name은 layer에서 만든 이름
    dM = maxabs_delta(out["M2"], Mref)
    dV = maxabs_delta(out["V2"], Vref)

    print("[OUT KEYS]", list(out.keys()))
    print(f"max|dP|={dP:.3e} max|dM|={dM:.3e} max|dV|={dV:.3e}")

    print("[OK]" if max(dP, dM, dV) == 0.0 else "[WARN] nonzero delta")


if __name__ == "__main__":
    main()
