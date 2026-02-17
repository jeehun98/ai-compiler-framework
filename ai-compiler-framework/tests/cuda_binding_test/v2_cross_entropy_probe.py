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
def schema_id_XENT() -> int:
    return int.from_bytes(b"XENT", "little", signed=False)

def pack_xent(ignore_index: int = -100, reduction: int = 0) -> bytes:
    return struct.pack("<ii", int(ignore_index), int(reduction))

def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())

def measure_time(func, rep=200, warmup=20):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        func()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep

def print_sanity(tag: str, logits: torch.Tensor, targets_i32: torch.Tensor, C: int, ignore_index: int):
    # targets_i32 must be int32
    tmin = int(targets_i32.min().item())
    tmax = int(targets_i32.max().item())
    ign = int((targets_i32 == int(ignore_index)).sum().item())
    bad = int(((targets_i32 != int(ignore_index)) & ((targets_i32 < 0) | (targets_i32 >= C))).sum().item())

    print(f"  [{tag}] logits finite? {bool(torch.isfinite(logits).all().item())} "
          f"| targets min/max {tmin}/{tmax} | ignore {ign} | OOR(non-ign) {bad}")

# -----------------------------------------------------------------------------
# Reference (PyTorch)
# -----------------------------------------------------------------------------
def cross_entropy_ref_loss(logits: torch.Tensor,
                           targets_i64: torch.Tensor,
                           ignore_index: int = -100,
                           reduction: int = 0) -> torch.Tensor:
    red = "mean" if reduction == 0 else "sum"
    return torch.nn.functional.cross_entropy(
        logits, targets_i64, ignore_index=ignore_index, reduction=red
    )

def cross_entropy_ref_dlogits(logits: torch.Tensor,
                              targets_i64: torch.Tensor,
                              ignore_index: int = -100,
                              reduction: int = 0) -> torch.Tensor:
    logits2 = logits.detach().clone().requires_grad_(True)
    loss = cross_entropy_ref_loss(logits2, targets_i64, ignore_index, reduction)
    loss.backward()
    return logits2.grad

# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_fwd(shape, C: int, ignore_index: int, reduction: int, do_bench=True, debug=True):
    device = torch.device("cuda:0")
    dtype = torch.float32

    N = int(shape[0])
    assert shape == (N, C)

    logits = torch.randn(N, C, device=device, dtype=dtype).contiguous()
    targets = torch.randint(0, C, (N,), device=device, dtype=torch.int32).contiguous()

    if ignore_index is not None:
        mask = torch.rand(N, device=device) < 0.1
        targets = targets.clone()
        targets[mask] = int(ignore_index)

    out_loss = torch.empty((1,), device=device, dtype=torch.float32).contiguous()

    if debug:
        print_sanity("FWD-in", logits, targets, C, ignore_index)

    ref = cross_entropy_ref_loss(
        logits, targets.to(torch.int64), ignore_index=int(ignore_index), reduction=int(reduction)
    ).view(1)

    def _run():
        _C.op_call(
            int(_C.OpKind.CrossEntropyFwd),
            [logits, targets],
            [out_loss],
            schema_id_XENT(),
            pack_xent(ignore_index=int(ignore_index), reduction=int(reduction)),
            0,
        )

    _run()

    if debug:
        print(f"  [FWD-out] out_loss={float(out_loss.item())} finite? {bool(torch.isfinite(out_loss).all().item())} "
              f"| ref={float(ref.item())} finite? {bool(torch.isfinite(ref).all().item())}")

    d = maxabs_delta(out_loss, ref)

    tag_red = "mean" if reduction == 0 else "sum"
    msg = f"[FWD-{tag_red}] Shape={(N, C)} | ignore={ignore_index} | dLoss={d:.2e}"

    if do_bench:
        ms = measure_time(_run, rep=200, warmup=20)
        msg += f" | {ms:.3f} ms"
    print(msg)
    return d

def run_bwd(shape, C: int, ignore_index: int, reduction: int, do_bench=True, debug=True):
    device = torch.device("cuda:0")
    dtype = torch.float32

    N = int(shape[0])
    assert shape == (N, C)

    logits = torch.randn(N, C, device=device, dtype=dtype).contiguous()
    targets = torch.randint(0, C, (N,), device=device, dtype=torch.int32).contiguous()

    if ignore_index is not None:
        mask = torch.rand(N, device=device) < 0.1
        targets = targets.clone()
        targets[mask] = int(ignore_index)

    grad_loss = torch.ones((1,), device=device, dtype=torch.float32).contiguous()
    dlogits = torch.empty_like(logits).contiguous()

    if debug:
        print_sanity("BWD-in", logits, targets, C, ignore_index)

    dref = cross_entropy_ref_dlogits(
        logits, targets.to(torch.int64), ignore_index=int(ignore_index), reduction=int(reduction)
    )

    def _run():
        _C.op_call(
            int(_C.OpKind.CrossEntropyBwd),
            [logits, targets, grad_loss],
            [dlogits],
            schema_id_XENT(),
            pack_xent(ignore_index=int(ignore_index), reduction=int(reduction)),
            0,
        )

    _run()

    if debug:
        fin = bool(torch.isfinite(dlogits).all().item())
        fin_ref = bool(torch.isfinite(dref).all().item())
        print(f"  [BWD-out] dlogits finite? {fin} | dref finite? {fin_ref} "
              f"| max|dlogits|={float(dlogits.abs().max().item())} "
              f"| max|dref|={float(dref.abs().max().item())}")

    d = maxabs_delta(dlogits, dref)

    tag_red = "mean" if reduction == 0 else "sum"
    msg = f"[BWD-{tag_red}] Shape={(N, C)} | ignore={ignore_index} | dGrad={d:.2e}"

    if do_bench:
        ms = measure_time(_run, rep=200, warmup=20)
        msg += f" | {ms:.3f} ms"
    print(msg)
    return d

def main():
    torch.manual_seed(0)
    print("CrossEntropyFwd enum value =", int(_C.OpKind.CrossEntropyFwd))
    print("CrossEntropyBwd enum value =", int(_C.OpKind.CrossEntropyBwd))
    print("-" * 100)

    # Small sanity
    C = 1024
    run_fwd((256, C), C=C, ignore_index=-100, reduction=0, do_bench=True, debug=True)
    run_bwd((256, C), C=C, ignore_index=-100, reduction=0, do_bench=True, debug=True)

    # Sum reduction
    run_fwd((256, C), C=C, ignore_index=-100, reduction=1, do_bench=True, debug=True)
    run_bwd((256, C), C=C, ignore_index=-100, reduction=1, do_bench=True, debug=True)

    print("-" * 100)

    # Larger (more realistic)
    C2 = 4096
    run_fwd((1024, C2), C=C2, ignore_index=-100, reduction=0, do_bench=True, debug=False)
    run_bwd((1024, C2), C=C2, ignore_index=-100, reduction=0, do_bench=True, debug=False)

    print("-" * 100)

    # Negative test: wrong dtype (targets int64 instead of int32)
    try:
        device = torch.device("cuda:0")
        logits = torch.randn(32, 128, device=device, dtype=torch.float32).contiguous()
        targets_bad = torch.randint(0, 128, (32,), device=device, dtype=torch.int64).contiguous()
        out = torch.empty((1,), device=device, dtype=torch.float32).contiguous()
        _C.op_call(
            int(_C.OpKind.CrossEntropyFwd),
            [logits, targets_bad],
            [out],
            schema_id_XENT(),
            pack_xent(-100, 0),
            0,
        )
        print("[NEG] expected failure but op_call succeeded (targets int64).")
    except Exception as e:
        print("[NEG] targets int64 -> expected fail:", str(e))

if __name__ == "__main__":
    main()
