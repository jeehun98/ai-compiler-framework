from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PYMOD_DIR  = REPO_ROOT / "build" / "python"
PKG_DIR    = PYMOD_DIR / "aicf_cuda"

sys.path.insert(0, str(PYMOD_DIR))
if os.name == "nt":
    os.add_dll_directory(str(PYMOD_DIR))
    os.add_dll_directory(str(PKG_DIR))

import torch
import aicf_cuda as aicf


def op_call(kind, inputs, outputs, attrs=None):
    if attrs is None:
        attrs = {}
    aicf.op_call(kind, inputs, outputs, attrs)


def check(name: str, got: torch.Tensor, ref: torch.Tensor, atol=1e-3, rtol=1e-3):
    max_abs = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[{name}] ok={ok} max_abs={max_abs} shape={tuple(got.shape)} dtype={got.dtype}")
    if not ok:
        raise RuntimeError(f"{name} mismatch (max_abs={max_abs})")


@torch.inference_mode()
def test_bias_add_f32():
    # Y: [M,N], bias: [N]
    M, N = 1234, 257
    Y = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(N, device="cuda", dtype=torch.float32).contiguous()
    O = torch.empty_like(Y)

    for _ in range(10):
        op_call(aicf.OpKind.BiasAdd, [Y, B], [O], {"axis": -1})
    torch.cuda.synchronize()

    ref = (Y + B.view(1, -1)).contiguous()
    check("BiasAdd f32 axis=-1", O, ref, atol=1e-5, rtol=1e-5)


@torch.inference_mode()
def test_bias_add_f16_scalar():
    M, N = 777, 513  # 홀수 N도 scalar path는 지원
    Y = torch.randn(M, N, device="cuda", dtype=torch.float16).contiguous()
    B = torch.randn(N, device="cuda", dtype=torch.float16).contiguous()
    O = torch.empty_like(Y)

    for _ in range(10):
        op_call(aicf.OpKind.BiasAdd, [Y, B], [O], {"axis": -1})
    torch.cuda.synchronize()

    ref = (Y + B.view(1, -1)).contiguous()
    # fp16이라 허용오차 조금
    check("BiasAdd f16 scalar axis=-1", O, ref, atol=3e-3, rtol=3e-3)


@torch.inference_mode()
def test_bias_add_f16_half2():
    # half2 path: last dim even + 4B align(대부분 contiguous면 만족)
    M, N = 1024, 1024
    Y = torch.randn(M, N, device="cuda", dtype=torch.float16).contiguous()
    B = torch.randn(N, device="cuda", dtype=torch.float16).contiguous()
    O = torch.empty_like(Y)

    for _ in range(10):
        op_call(aicf.OpKind.BiasAdd, [Y, B], [O], {"axis": -1})
    torch.cuda.synchronize()

    ref = (Y + B.view(1, -1)).contiguous()
    check("BiasAdd f16 half2 axis=-1", O, ref, atol=3e-3, rtol=3e-3)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.cuda.synchronize()

    test_bias_add_f32()
    test_bias_add_f16_scalar()
    test_bias_add_f16_half2()

    print("ALL OK")
