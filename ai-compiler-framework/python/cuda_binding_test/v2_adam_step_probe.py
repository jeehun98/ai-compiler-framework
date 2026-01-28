from __future__ import annotations
import sys
from pathlib import Path
import struct
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]
EX_PY = ROOT / "examples" / "python"
BUILD_PY = ROOT / "build" / "python"
for p in (EX_PY, BUILD_PY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from aicf_cuda import _C


def schema_id_ADAM() -> int:
    # 'ADAM' little-endian -> 0x4D414441
    return int.from_bytes(b"ADAM", "little", signed=False)


def pack_adam(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8) -> bytes:
    return struct.pack("<ffff", float(lr), float(beta1), float(beta2), float(eps))


def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def adam_ref(P, G, M, V, bc1, bc2, lr, beta1, beta2, eps):
    m_new = beta1 * M + (1.0 - beta1) * G
    v_new = beta2 * V + (1.0 - beta2) * (G * G)
    m_hat = m_new / bc1
    v_hat = v_new / bc2
    P_new = P - lr * (m_hat / (torch.sqrt(v_hat) + eps))
    return P_new, m_new, v_new


def run(shape, inplace: bool):
    device = torch.device("cuda:0")
    dtype = torch.float32

    lr, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8

    P = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    G = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    M = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    V = torch.randn(*shape, device=device, dtype=dtype).contiguous()

    # bias-correction scalars (rank0)
    bc1 = torch.tensor(1.0 - beta1**3, device=device, dtype=dtype).contiguous()
    bc2 = torch.tensor(1.0 - beta2**3, device=device, dtype=dtype).contiguous()

    if inplace:
        Pout = P
        Mout = M
        Vout = V
    else:
        Pout = P.clone()
        Mout = torch.empty_like(M)
        Vout = torch.empty_like(V)

    Pref, Mref, Vref = adam_ref(P, G, M, V, bc1, bc2, lr, beta1, beta2, eps)

    _C.op_call(
        int(_C.OpKind.AdamStep),
        [P, G, M, V, bc1, bc2],
        [Pout, Mout, Vout],
        schema_id_ADAM(),
        pack_adam(lr, beta1, beta2, eps),
        0,
    )

    dP = maxabs_delta(Pout, Pref)
    dM = maxabs_delta(Mout, Mref)
    dV = maxabs_delta(Vout, Vref)
    tag = "inplace" if inplace else "oop"
    print(f"[F32 {tag}] shape={tuple(shape)} max|dP|={dP:.3e} max|dM|={dM:.3e} max|dV|={dV:.3e}")
    return max(dP, dM, dV)


def main():
    torch.manual_seed(0)
    print("AdamStep enum value =", int(_C.OpKind.AdamStep))
    print(f"schema_id(ADAM) = 0x{schema_id_ADAM():08X}")

    worst = 0.0
    for shape in [(1024,), (64, 256), (8, 32, 128)]:
        worst = max(worst, run(shape, inplace=False))
    worst = max(worst, run((64, 256), inplace=True))
    print("[F32] worst max delta =", worst)

    # NEG: dtype mismatch
    P = torch.randn(128, device="cuda", dtype=torch.float16).contiguous()
    G = torch.randn(128, device="cuda", dtype=torch.float16).contiguous()
    M = torch.randn(128, device="cuda", dtype=torch.float16).contiguous()
    V = torch.randn(128, device="cuda", dtype=torch.float16).contiguous()
    bc1 = torch.tensor(0.1, device="cuda", dtype=torch.float32).contiguous()
    bc2 = torch.tensor(0.1, device="cuda", dtype=torch.float32).contiguous()
    Pout = P.clone()
    Mout = torch.empty_like(M)
    Vout = torch.empty_like(V)
    try:
        _C.op_call(int(_C.OpKind.AdamStep), [P, G, M, V, bc1, bc2], [Pout, Mout, Vout],
                   schema_id_ADAM(), pack_adam(), 0)
        print("[NEG dtype] unexpected OK")
    except RuntimeError as e:
        print("[NEG dtype] ok:", str(e).splitlines()[0])

    # NEG: alias grad (G==Pout) should be InvalidArgument
    P = torch.randn(128, device="cuda", dtype=torch.float32).contiguous()
    G = torch.randn(128, device="cuda", dtype=torch.float32).contiguous()
    M = torch.randn(128, device="cuda", dtype=torch.float32).contiguous()
    V = torch.randn(128, device="cuda", dtype=torch.float32).contiguous()
    bc1 = torch.tensor(0.1, device="cuda", dtype=torch.float32).contiguous()
    bc2 = torch.tensor(0.1, device="cuda", dtype=torch.float32).contiguous()
    try:
        _C.op_call(int(_C.OpKind.AdamStep), [P, G, M, V, bc1, bc2], [G, M, V],
                   schema_id_ADAM(), pack_adam(), 0)
        print("[NEG alias grad] unexpected OK")
    except RuntimeError as e:
        print("[NEG alias grad] ok:", str(e).splitlines()[0])


if __name__ == "__main__":
    main()
