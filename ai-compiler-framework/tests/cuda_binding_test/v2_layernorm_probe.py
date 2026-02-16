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

def schema_id(tag4: str) -> int:
    b = tag4.encode("ascii")
    assert len(b) == 4
    return int.from_bytes(b, "little", signed=False)

def measure_bandwidth(func, *args, N, M, dtype_size, mode="fwd", rep=100, warmup=10):
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
    
    total_elements = M * N
    
    if mode == "fwd":
        # Read X, Write Y (Mean/Var are negligible)
        # Note: Kernel reads X twice? Ideally L2 cached. We count DRAM traffic as 2x.
        factor = 2
    elif mode == "bwd":
        # Pass 1 (dX): Read X, dY, Write dX (3x)
        # Pass 2 (dG, dB): Read X, dY (2x)
        factor = 5
    else:
        factor = 0

    total_bytes = total_elements * dtype_size * factor
    gbps = total_bytes / (avg_ms / 1000.0) / 1e9
    return avg_ms, gbps

def run_bench(dtype, M, N, affine=True):
    device = torch.device("cuda:0")
    dtype_size = 2 if dtype == torch.float16 else 4
    eps = 1e-5
    
    x = torch.randn(M, N, device=device, dtype=dtype)
    dy = torch.randn(M, N, device=device, dtype=dtype)
    
    sid = schema_id("LNEP")
    payload = struct.pack("<f", float(eps))
    
    # -------------------------------------------------------
    # Forward Bench
    # -------------------------------------------------------
    if affine:
        g = torch.randn(N, device=device, dtype=dtype)
        b = torch.randn(N, device=device, dtype=dtype)
        inputs_fwd = [x, g, b]
    else:
        inputs_fwd = [x]
        
    y = torch.empty_like(x)
    mean = torch.empty(M, device="cuda", dtype=torch.float32)
    rstd = torch.empty(M, device="cuda", dtype=torch.float32)
    outputs_fwd = [y, mean, rstd]

    def _run_fwd():
        _C.op_call(_C.OpKind.LayerNormFwd, inputs_fwd, outputs_fwd, sid, payload, 0)

    ms_fwd, gbps_fwd = measure_bandwidth(_run_fwd, N=N, M=M, dtype_size=dtype_size, mode="fwd")

    # -------------------------------------------------------
    # Backward Bench
    # -------------------------------------------------------
    if affine:
        dx = torch.empty_like(x)
        dgamma = torch.empty(N, device="cuda", dtype=torch.float32)
        dbeta  = torch.empty(N, device="cuda", dtype=torch.float32)
        # inputs: X, dY, G, Mean, Rstd
        inputs_bwd = [x, dy, g, mean, rstd]
        outputs_bwd = [dx, dgamma, dbeta]
    else:
        dx = torch.empty_like(x)
        # inputs: X, dY, Mean, Rstd
        inputs_bwd = [x, dy, mean, rstd]
        outputs_bwd = [dx]

    def _run_bwd():
        _C.op_call(_C.OpKind.LayerNormBwd, inputs_bwd, outputs_bwd, sid, payload, 0)
        
    ms_bwd, gbps_bwd = measure_bandwidth(_run_bwd, N=N, M=M, dtype_size=dtype_size, mode="bwd")

    # Print
    name = f"{'F16' if dtype==torch.float16 else 'F32'}"
    print(f"[{name}] {M}x{N:<5} | FWD: {ms_fwd:.3f} ms ({gbps_fwd:.1f} GB/s) | BWD: {ms_bwd:.3f} ms ({gbps_bwd:.1f} GB/s)")

def main():
    torch.manual_seed(0)
    print(f"LayerNorm Benchmark (Target: >250 GB/s)")
    print("-" * 80)

    # Large shape to saturate GPU
    # M=4096, N=4096 => 16M elements
    M, N = 4096, 4096
    
    run_bench(torch.float32, M, N, affine=True)
    run_bench(torch.float16, M, N, affine=True)
    
    print("-" * 80)
    # Check "Hidden Size" typical shapes (M large, N small)
    # BERT-base style: N=768
    M_bert, N_bert = 32*1024, 768 # Total ~25M elements
    run_bench(torch.float16, M_bert, N_bert, affine=True)

if __name__ == "__main__":
    main()