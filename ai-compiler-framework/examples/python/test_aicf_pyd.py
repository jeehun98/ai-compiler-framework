import os
import sys
from pathlib import Path

# 이 파일(examples/python/test_aicf_pyd.py) 기준으로 repo 루트 추정
REPO_ROOT = Path(__file__).resolve().parents[2]          # examples/python -> repo
PYMOD_DIR  = REPO_ROOT / "build" / "python"              # build/python
PKG_DIR    = PYMOD_DIR / "aicf_cuda"                     # build/python/aicf_cuda

# 파이썬 모듈 탐색 경로에 build/python 추가
sys.path.insert(0, str(PYMOD_DIR))

# (Windows) DLL 로드 실패 대비: pyd가 의존 dll을 같은 폴더에서 찾게 함
if os.name == "nt":
    os.add_dll_directory(str(PYMOD_DIR))
    os.add_dll_directory(str(PKG_DIR))

import torch

import aicf_cuda.add as aicf_add
import aicf_cuda.relu as aicf_relu
import aicf_cuda.gemm as aicf_gemm

def check(name, got, ref, atol=1e-4, rtol=1e-4):
    max_abs = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[{name}] ok={ok} max_abs={max_abs}")
    if not ok:
        raise RuntimeError(f"{name} mismatch")


def test_add():
    a = torch.randn(1024, device="cuda", dtype=torch.float32)
    b = torch.randn(1024, device="cuda", dtype=torch.float32)
    out = torch.empty_like(a)

    ok = aicf_add.add_f32(a, b, out)
    assert ok

    ref = a + b
    check("add_f32", out, ref)


def test_relu():
    x = torch.randn(2048, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    ok = aicf_relu.relu_f32(x, out)
    assert ok

    ref = torch.relu(x)
    check("relu_f32", out, ref)


def test_gemm():
    M, K, N = 128, 256, 96
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)

    ok = aicf_gemm.gemm_f32(A, B, C)
    assert ok

    ref = A @ B
    check("gemm_f32", C, ref, atol=2e-3, rtol=2e-3)  # naive float 누적이라 오차 조금 허용


if __name__ == "__main__":
    assert torch.cuda.is_available()
    test_add()
    test_relu()
    test_gemm()
    print("ALL OK")

# ncu --target-processes all python .\test_aicf_pyd.py
