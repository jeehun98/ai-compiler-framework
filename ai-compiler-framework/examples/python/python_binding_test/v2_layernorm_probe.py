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

import struct
import torch
import aicf_cuda._C as _C

def schema_id(tag4: str) -> int:
    b = tag4.encode("ascii")
    assert len(b) == 4
    return int.from_bytes(b, "little", signed=False)

def ref_layernorm_fwd(x, gamma=None, beta=None, eps=1e-5):
    mu = x.mean(dim=1)
    var = x.var(dim=1, unbiased=False)
    rstd = (var + eps).rsqrt()
    xhat = (x - mu[:, None]) * rstd[:, None]
    if gamma is not None and beta is not None:
        y = xhat * gamma[None, :] + beta[None, :]
    else:
        y = xhat
    return y, mu.to(torch.float32), rstd.to(torch.float32)

def ref_layernorm_bwd(x, dy, mean, rstd, gamma=None):
    M, N = x.shape
    mu = mean.to(x.dtype)
    rs = rstd.to(x.dtype)
    xhat = (x - mu[:, None]) * rs[:, None]
    dy_hat = dy if gamma is None else (dy * gamma[None, :])

    s1 = dy_hat.sum(dim=1)
    s2 = (dy_hat * xhat).sum(dim=1)
    dx = ((N * dy_hat - s1[:, None] - xhat * s2[:, None]) * (rs[:, None] / N))

    if gamma is None:
        return dx, None, None
    dgamma = (dy * xhat).sum(dim=0).to(torch.float32)
    dbeta  = dy.sum(dim=0).to(torch.float32)
    return dx, dgamma, dbeta

def call_op(kind, inputs, outputs, sid=0, payload=b"", stream=0):
    # signature: (kind, inputs, outputs, schema_id:int=0, attrs_bytes:bytes=b'', stream:int=0)
    _C.op_call(kind, inputs, outputs, sid, payload, stream)

def run_fwd(dtype=torch.float32, affine=False, M=8, N=128, eps=1e-5):
    x = torch.randn(M, N, device="cuda", dtype=dtype)

    sid = schema_id("LNEP")
    payload = struct.pack("<f", float(eps))

    if affine:
        g = torch.randn(N, device="cuda", dtype=dtype)
        b = torch.randn(N, device="cuda", dtype=dtype)
        y = torch.empty_like(x)
        mean = torch.empty(M, device="cuda", dtype=torch.float32)
        rstd = torch.empty(M, device="cuda", dtype=torch.float32)

        call_op(_C.OpKind.LayerNormFwd, [x, g, b], [y, mean, rstd], sid, payload)

        y_ref, mean_ref, rstd_ref = ref_layernorm_fwd(x, g, b, eps)
    else:
        y = torch.empty_like(x)
        mean = torch.empty(M, device="cuda", dtype=torch.float32)
        rstd = torch.empty(M, device="cuda", dtype=torch.float32)

        call_op(_C.OpKind.LayerNormFwd, [x], [y, mean, rstd], sid, payload)

        y_ref, mean_ref, rstd_ref = ref_layernorm_fwd(x, None, None, eps)

    return max(
        (y - y_ref).abs().max().item(),
        (mean - mean_ref).abs().max().item(),
        (rstd - rstd_ref).abs().max().item(),
    )

def run_bwd(dtype=torch.float32, affine=False, M=8, N=128, eps=1e-5):
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    dy = torch.randn(M, N, device="cuda", dtype=dtype)

    if affine:
        g = torch.randn(N, device="cuda", dtype=dtype)
        b = torch.randn(N, device="cuda", dtype=dtype)
        _, mean_ref, rstd_ref = ref_layernorm_fwd(x, g, b, eps)
    else:
        _, mean_ref, rstd_ref = ref_layernorm_fwd(x, None, None, eps)
        g = None

    mean = mean_ref.contiguous()
    rstd = rstd_ref.contiguous()

    if affine:
        dx = torch.empty_like(x)
        dgamma = torch.empty(N, device="cuda", dtype=torch.float32)
        dbeta  = torch.empty(N, device="cuda", dtype=torch.float32)

        call_op(_C.OpKind.LayerNormBwd, [x, dy, g, mean, rstd], [dx, dgamma, dbeta])

        dx_ref, dgamma_ref, dbeta_ref = ref_layernorm_bwd(x, dy, mean, rstd, g)
        return max(
            (dx - dx_ref).abs().max().item(),
            (dgamma - dgamma_ref).abs().max().item(),
            (dbeta - dbeta_ref).abs().max().item(),
        )
    else:
        dx = torch.empty_like(x)
        call_op(_C.OpKind.LayerNormBwd, [x, dy, mean, rstd], [dx])

        dx_ref, _, _ = ref_layernorm_bwd(x, dy, mean, rstd, None)
        return (dx - dx_ref).abs().max().item()

def main():
    print("LayerNormFwd enum value =", int(_C.OpKind.LayerNormFwd))
    print("LayerNormBwd enum value =", int(_C.OpKind.LayerNormBwd))
    print("schema_id(LNEP) =", hex(schema_id("LNEP")))

    worst = 0.0
    for dtype in (torch.float32, torch.float16):
        for affine in (False, True):
            for (M, N) in ((8, 128), (64, 256), (7, 33)):
                d = run_fwd(dtype=dtype, affine=affine, M=M, N=N, eps=1e-5)
                print(f"[FWD {dtype} affine={affine}] M={M} N={N} max|d|={d:.3e}")
                worst = max(worst, d)

    for dtype in (torch.float32, torch.float16):
        for affine in (False, True):
            for (M, N) in ((8, 128), (64, 256), (7, 33)):
                d = run_bwd(dtype=dtype, affine=affine, M=M, N=N, eps=1e-5)
                print(f"[BWD {dtype} affine={affine}] M={M} N={N} max|d|={d:.3e}")
                worst = max(worst, d)

    print("[OK] worst max|delta| =", worst)

    # NEG: wrong rank (3D) -> expect NotImplemented
    try:
        x = torch.randn(2, 3, 4, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        mean = torch.empty(2, device="cuda", dtype=torch.float32)
        rstd = torch.empty(2, device="cuda", dtype=torch.float32)
        sid = schema_id("LNEP")
        payload = struct.pack("<f", 1e-5)
        call_op(_C.OpKind.LayerNormFwd, [x], [y, mean, rstd], sid, payload)
        raise AssertionError("expected failure")
    except RuntimeError as e:
        print("[NEG rank] ok:", str(e).splitlines()[0])

if __name__ == "__main__":
    main()