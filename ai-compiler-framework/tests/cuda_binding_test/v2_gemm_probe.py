from __future__ import annotations
import sys
from pathlib import Path
import torch
import struct

THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]
EX_PY = ROOT / "examples" / "python"
BUILD_PY = ROOT / "build" / "python"
for p in (EX_PY, BUILD_PY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
        
import _C

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def pack_gemm(trans_a: int = 0, trans_b: int = 0) -> bytes:
    return struct.pack("<ii", int(trans_a), int(trans_b))

def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())

def make_A_B(M: int, K: int, N: int, ta: bool, tb: bool, dtype):
    device = torch.device("cuda:0")
    if not ta:
        A = torch.randn(M, K, device=device, dtype=dtype).contiguous()
    else:
        A = torch.randn(K, M, device=device, dtype=dtype).contiguous()

    if not tb:
        B = torch.randn(K, N, device=device, dtype=dtype).contiguous()
    else:
        B = torch.randn(N, K, device=device, dtype=dtype).contiguous()
    return A, B

def torch_gemm_ref(A: torch.Tensor, B: torch.Tensor, ta: bool, tb: bool):
    A2 = A.t() if ta else A
    B2 = B.t() if tb else B
    return A2 @ B2

# -------------------------------------------------------------------------
# Benchmark Helper
# -------------------------------------------------------------------------
def measure_performance(func, *args, M, N, K, rep=100, warmup=10):
    """
    Runs the kernel multiple times and measures execution time using CUDA Events.
    Returns: (avg_time_ms, tflops)
    """
    # 1. Warmup
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    # 2. Measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        func(*args)
    end_event.record()
    
    torch.cuda.synchronize()
    
    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / rep
    
    # TFLOPS = (2 * M * N * K) / (seconds * 10^12)
    # seconds = avg_ms / 1000
    ops = 2.0 * M * N * K
    tflops = (ops / (avg_ms / 1000.0)) / 1e12
    
    return avg_ms, tflops

# -------------------------------------------------------------------------
# Runners
# -------------------------------------------------------------------------

def run_f32(M, K, N, ta=False, tb=False, do_bench=True):
    A, B = make_A_B(M, K, N, ta, tb, torch.float32)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()
    
    # 1. Correctness Check
    C_ref = torch_gemm_ref(A, B, ta, tb)
    
    # Kernel Call Wrapper
    def _run_kernel():
        _C.op_call(
            int(_C.OpKind.Gemm),
            [A, B],
            [C],
            0,
            pack_gemm(ta, tb),
            0,
        )

    _run_kernel() # Run once for check
    d = maxabs_delta(C, C_ref)
    
    msg = f"[F32   ] ({M}x{N}x{K}) ta={int(ta)} tb={int(tb)} | Diff={d:.1e}"

    # 2. Benchmark
    if do_bench:
        ms, tflops = measure_performance(_run_kernel, M=M, N=N, K=K)
        msg += f" | Time={ms:.3f} ms | Perf={tflops:.4f} TFLOPS"
        
    print(msg)
    return d

def run_f16_tc(M, K, N, ta=False, tb=False, do_bench=True):
    A, B = make_A_B(M, K, N, ta, tb, torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16).contiguous()
    
    # 1. Correctness Check
    C_ref = torch_gemm_ref(A.float(), B.float(), ta, tb).half()
    
    # Kernel Call Wrapper
    def _run_kernel():
        _C.op_call(
            int(_C.OpKind.Gemm),
            [A, B],
            [C],
            0,
            pack_gemm(ta, tb),
            0,
        )

    _run_kernel()
    d = maxabs_delta(C, C_ref)
    
    msg = f"[F16-TC] ({M}x{N}x{K}) ta={int(ta)} tb={int(tb)} | Diff={d:.1e}"

    # 2. Benchmark
    if do_bench:
        ms, tflops = measure_performance(_run_kernel, M=M, N=N, K=K)
        msg += f" | Time={ms:.3f} ms | Perf={tflops:.4f} TFLOPS"

    print(msg)
    return d

def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    print(f"Gemm enum value = {int(_C.OpKind.Gemm)}")
    print("-" * 80)
    print("Kernel   ( M x N x K ) Flags         Diff      Time       Performance")
    print("-" * 80)

    # Size configurations (M, K, N)
    # Some larger sizes to see performance better
    sizes = [
        (64, 48, 80),      # Small
        (1024, 1024, 1024) # Medium (Standard Benchmark)
    ]

    for M, K, N in sizes:
        # F32 Test
        for ta in (False, True):
            for tb in (False, True):
                run_f32(M, K, N, ta, tb)
        print("-" * 20)
        
        # F16 TC Test
        # WMMA usually prefers larger multiples of 16/32
        M_tc, K_tc, N_tc = (x if x >= 64 else 64 for x in (M, K, N)) 
        
        for ta in (False, True):
            for tb in (False, True):
                run_f16_tc(M_tc, K_tc, N_tc, ta, tb)
        print("=" * 80)

    # NEG Test (unchanged)
    print("\n[Negative Test: Invalid Shape]")
    A, B = make_A_B(8, 4, 7, False, False, torch.float32)
    C_bad = torch.empty(8, 8, device="cuda", dtype=torch.float32).contiguous()
    try:
        _C.op_call(int(_C.OpKind.Gemm), [A, B], [C_bad], 0, pack_gemm(0, 0), 0)
        print("Unexpected OK")
    except RuntimeError as e:
        print(f"Caught expected error: {str(e).splitlines()[0]}")

if __name__ == "__main__":
    main()