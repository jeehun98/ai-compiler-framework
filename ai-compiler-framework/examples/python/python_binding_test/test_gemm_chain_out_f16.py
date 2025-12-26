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


def check(name, got_f16, ref_f32, atol=1e-1, rtol=1e-1):
    got = got_f16.float()
    ref = ref_f32.float()
    max_abs = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[{name}] ok={ok} max_abs={max_abs} got.dtype={got_f16.dtype}")
    if not ok:
        raise RuntimeError(f"{name} mismatch")


@torch.inference_mode()
def test_linear_bias_relu_chain(iters=200):
    M, K, N = 256, 256, 256

    # X[M,K], W[K,N], b[N]
    X = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    W = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    b = torch.randn(N, device="cuda", dtype=torch.float16).contiguous()

    # Z = X @ W  (out_f16)
    Z = torch.empty(M, N, device="cuda", dtype=torch.float16).contiguous()
    # Y = Z + b  (out_f16)
    Y = torch.empty(M, N, device="cuda", dtype=torch.float16).contiguous()
    # O = relu(Y) (out_f16)
    O = torch.empty(M, N, device="cuda", dtype=torch.float16).contiguous()

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.Gemm, [X, W], [Z], {"transA": False, "transB": False})
        op_call(aicf.OpKind.BiasAdd, [Z, b], [Y], {})
        op_call(aicf.OpKind.EltwiseRelu, [Y], [O], {})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::linear_bias_relu_chain"):
        for _ in range(iters):
            op_call(aicf.OpKind.Gemm, [X, W], [Z], {"transA": False, "transB": False})
            op_call(aicf.OpKind.BiasAdd, [Z, b], [Y], {})
            op_call(aicf.OpKind.EltwiseRelu, [Y], [O], {})
    torch.cuda.synchronize()

    # torch ref (f32 for accuracy)
    ref = torch.relu(X.float() @ W.float() + b.float()).contiguous()
    check("Chain(Gemm out_f16 + BiasAdd + Relu)", O, ref)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.cuda.synchronize()
    test_linear_bias_relu_chain()
    print("ALL OK")
