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

import _C

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def schema_id_ADAM() -> int:
    return int.from_bytes(b"ADAM", "little", signed=False)

def pack_adam(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8) -> bytes:
    return struct.pack("<ffff", float(lr), float(beta1), float(beta2), float(eps))

def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())

def measure_bandwidth(func, *args, numel, rep=100, warmup=10):
    """
    Adam Memory Access:
    Read: P, G, M, V (4 arrays)
    Write: P, M, V (3 arrays)
    Total: 7 * 4 bytes * numel
    """
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        func(*args)
    end_event.record()
    
    torch.cuda.synchronize()
    avg_ms = start_event.elapsed_time(end_event) / rep
    
    # 7 accesses per element (Read P,G,M,V + Write P,M,V) * 4 bytes
    total_bytes = numel * 4 * 7
    gbps = total_bytes / (avg_ms / 1000.0) / 1e9
    return avg_ms, gbps

# -----------------------------------------------------------------------------
# Reference
# -----------------------------------------------------------------------------
def adam_ref(P, G, M, V, bc1, bc2, lr, beta1, beta2, eps):
    m_new = beta1 * M + (1.0 - beta1) * G
    v_new = beta2 * V + (1.0 - beta2) * (G * G)
    m_hat = m_new / bc1
    v_hat = v_new / bc2
    P_new = P - lr * (m_hat / (torch.sqrt(v_hat) + eps))
    return P_new, m_new, v_new

# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run(shape, inplace: bool, do_bench=True):
    device = torch.device("cuda:0")
    dtype = torch.float32

    lr, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8

    P = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    G = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    M = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    
    # 🔥 [Fix] V must be non-negative!
    V = torch.abs(torch.randn(*shape, device=device, dtype=dtype)).contiguous()

    # bias-correction scalars
    bc1 = torch.tensor(1.0 - beta1**3, device=device, dtype=dtype).contiguous()
    bc2 = torch.tensor(1.0 - beta2**3, device=device, dtype=dtype).contiguous()

    if inplace:
        Pout, Mout, Vout = P, M, V
    else:
        Pout = P.clone()
        Mout = torch.empty_like(M)
        Vout = torch.empty_like(V)

    # 1. Correctness
    Pref, Mref, Vref = adam_ref(P, G, M, V, bc1, bc2, lr, beta1, beta2, eps)
    
    # Kernel Wrapper
    def _run():
        _C.op_call(
            int(_C.OpKind.AdamStep),
            [P, G, M, V, bc1, bc2],
            [Pout, Mout, Vout],
            schema_id_ADAM(),
            pack_adam(lr, beta1, beta2, eps),
            0,
        )

    _run() # Run once

    dP = maxabs_delta(Pout, Pref)
    dM = maxabs_delta(Mout, Mref)
    dV = maxabs_delta(Vout, Vref)
    
    tag = "InPlace" if inplace else "OOP"
    msg = f"[{tag:<7}] Shape={str(tuple(shape)):<16} | dP={dP:.1e} dM={dM:.1e} dV={dV:.1e}"

    # 2. Benchmark
    if do_bench:
        numel = P.numel()
        ms, gbps = measure_bandwidth(_run, numel=numel)
        msg += f" | {ms:.3f} ms | {gbps:.2f} GB/s"
    
    print(msg)
    return max(dP, dM, dV)

def main():
    torch.manual_seed(0)
    print("AdamStep enum value =", int(_C.OpKind.AdamStep))
    print("-" * 100)

    # 1. Warmup & Small tests
    run((1024,), inplace=False)
    
    # 2. Large Benchmark (16M elements -> ~448MB data movement)
    # 4096*4096 = 16,777,216
    large_shape = (4096, 4096)
    run(large_shape, inplace=True)
    run(large_shape, inplace=False)

    print("-" * 100)
    
    # Negative tests (Keep as is)
    # ... (기존 Negative Test 코드 유지) ...

if __name__ == "__main__":
    main()