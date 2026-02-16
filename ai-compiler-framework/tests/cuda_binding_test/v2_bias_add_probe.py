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

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def fourcc(s: str) -> int:
    assert len(s) == 4
    return int.from_bytes(s.encode("ascii"), "little")

SCHEMA_BADD = fourcc("BADD")

def pack_bias_add(axis: int = -1) -> bytes:
    return struct.pack("<q", int(axis))

def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())

def measure_bandwidth(func, *args, numel, dtype_size, rep=100, warmup=10):
    """
    BiasAdd Memory Pattern:
    - Read Y (Large)
    - Read Bias (Small, fits in L2 Cache -> Ignore DRAM traffic)
    - Write Out (Large)
    => Effective DRAM Access = 2 * numel
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
    
    # 2 accesses (Read Y + Write Out)
    total_bytes = numel * dtype_size * 2
    gbps = total_bytes / (avg_ms / 1000.0) / 1e9
    
    return avg_ms, gbps

# -----------------------------------------------------------------------------
# Runners
# -----------------------------------------------------------------------------
def run_test(shape, dtype, name, axis=-1, do_bench=True):
    device = torch.device("cuda:0")
    is_fp16 = (dtype == torch.float16)
    dtype_size = 2 if is_fp16 else 4
    
    Y = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    N_dim = shape[-1]
    bias = torch.randn(N_dim, device=device, dtype=dtype).contiguous()
    O = torch.empty_like(Y).contiguous()

    # Ref
    if is_fp16:
        ref_y = Y.float()
        ref_b = bias.float()
        ref = (ref_y + ref_b).half()
    else:
        ref = Y + bias

    # Kernel Wrapper
    def _run():
        _C.op_call(
            int(_C.OpKind.BiasAdd),
            [Y, bias],
            [O],
            SCHEMA_BADD,
            pack_bias_add(axis),
            0,
        )

    # Correctness
    _run()
    d = maxabs_delta(O, ref)
    
    msg = f"[{name:<9}] Shape={str(tuple(shape)):<20} | Diff={d:.1e}"
    
    # Benchmark
    if do_bench:
        numel = Y.numel()
        ms, gbps = measure_bandwidth(_run, numel=numel, dtype_size=dtype_size)
        msg += f" | {ms:.3f} ms | {gbps:.2f} GB/s"
        
    print(msg)
    return d

def main():
    torch.manual_seed(0)
    print(f"BiasAdd Enum = {int(_C.OpKind.BiasAdd)}")
    print("-" * 90)
    
    # 1. Large Benchmark (16M elements)
    # 4096 * 4096 = 16,777,216
    large_shape = (4096, 4096)
    
    # F32
    run_test(large_shape, torch.float32, "F32", axis=-1)
    
    print("-" * 90)
    
    # F16 Naive (Force odd size to hit naive path)
    odd_shape = (4096 * 4096 - 1,) # 1D flattened
    # BiasAdd supports 1D as long as rank>=2 is not enforced strictly or handled as [N] broadcast
    # But launcher code says "is_contig_rank_ge2". Let's use (4096, 4095) for Odd N
    odd_shape_2d = (4096, 4095) 
    run_test(odd_shape_2d, torch.float16, "F16-Naive", axis=-1)

    # F16 Vec2 (Even size -> hit half2 path)
    run_test(large_shape, torch.float16, "F16-Vec2", axis=-1)

    print("-" * 90)

    # Negative Cases
    print("[Negative Tests]")
    try:
        # Rank < 2
        Y = torch.randn(10, device="cuda")
        B = torch.randn(10, device="cuda")
        O = torch.empty_like(Y)
        _C.op_call(int(_C.OpKind.BiasAdd), [Y, B], [O], SCHEMA_BADD, pack_bias_add(-1), 0)
    except RuntimeError as e:
        print(f"[Rank Mismatch] OK: {str(e).splitlines()[0]}")

if __name__ == "__main__":
    main()