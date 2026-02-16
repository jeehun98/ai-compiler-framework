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

RSUM = 0x5253554D  # 'RSUM'
def pack_axis(axis: int) -> bytes:
    return struct.pack("<q", int(axis))

def measure_bandwidth(func, *args, numel, dtype_size, rep=100, warmup=10):
    """
    ReduceSum (Col-wise) Memory Pattern:
    - Read Input (M*N)
    - Write Output (N) -> Negligible if M >> 1
    => Traffic ≈ 1x Read
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
    
    # GB/s = (Input Size) / Time (Read-Only dominant)
    total_bytes = numel * dtype_size
    gbps = total_bytes / (avg_ms / 1000.0) / 1e9
    return avg_ms, gbps

def run_bench(M, N, dtype, name):
    device = torch.device("cuda:0")
    dtype_size = 2 if dtype == torch.float16 else 4
    
    dY = torch.randn(M, N, device=device, dtype=dtype).contiguous()
    dB = torch.empty((N,), device=device, dtype=(torch.float32 if dtype==torch.float32 else torch.float16)).contiguous()
    
    def _run():
        # axis=0 means "reduce leading dims, keep last dim N"
        _C.op_call(int(_C.OpKind.ReduceSum), [dY], [dB], RSUM, pack_axis(0), 0)

    numel = M * N
    ms, gbps = measure_bandwidth(_run, numel=numel, dtype_size=dtype_size)
    
    print(f"[{name:<10}] Input={M}x{N:<5} | {ms:.3f} ms | {gbps:.2f} GB/s")

def main():
    torch.manual_seed(0)
    print(f"ReduceSum (Col-wise) Benchmark")
    print("-" * 80)
    
    # Case 1: N=1 (Global Sum) -> Stride = 1 (Coalesced!)
    # 16M elements
    run_bench(16*1024*1024, 1, torch.float32, "F32-Global")
    
    # Case 2: N=32 (Small Vector) -> Stride = 32*4 = 128 bytes (Cache Line Boundary)
    run_bench(512*1024, 32, torch.float32, "F32-Vec32")

    # Case 3: N=1024 (Bias Grad) -> Stride = 4096 bytes (Bad)
    run_bench(16*1024, 1024, torch.float32, "F32-Bias")
    
    # Case 4: N=4096 (Large) -> Stride = 16KB (Very Bad)
    run_bench(4096, 4096, torch.float32, "F32-Large")

    print("-" * 80)
    
    # F16 Check
    run_bench(16*1024*1024, 1, torch.float16, "F16-Global") # Coalesced
    run_bench(16*1024, 1024, torch.float16, "F16-Bias")     # Strided

if __name__ == "__main__":
    main()