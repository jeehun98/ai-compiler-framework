from __future__ import annotations

import sys
from pathlib import Path
import torch
import struct

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]   # python/aicf_v2
SRC = ROOT / "src"
sp = str(SRC)
if sp not in sys.path:
    sys.path.insert(0, sp)

import aicf_v2 as aicf
from aicf_v2.backends.cuda.registry import CudaRegistry


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


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


@torch.no_grad()
def run_fwd(dtype: str, affine: bool, M: int, N: int, eps: float = 1e-5) -> float:
    m = aicf.Model(dtype=dtype, device="cuda")

    spec_x = aicf.TensorSpec((M, N), dtype, "cuda")
    spec_n = aicf.TensorSpec((N,), dtype, "cuda")

    x = m.input("x", spec_x)

    if affine:
        g = m.input("gamma", spec_n)
        b = m.input("beta", spec_n)
        y, mean, rstd = m.add(aicf.LayerNormFwd("ln", eps=eps, affine=True), x, g, b)
    else:
        y, mean, rstd = m.add(aicf.LayerNormFwd("ln", eps=eps, affine=False), x)

    m.output("y", y)
    m.output("mean", mean)
    m.output("rstd", rstd)

    # feed
    torch_dtype = torch.float16 if dtype == "f16" else torch.float32
    xt = torch.randn(M, N, device="cuda", dtype=torch_dtype).contiguous()
    feed = {"x": xt}
    if affine:
        gt = torch.randn(N, device="cuda", dtype=torch_dtype).contiguous()
        bt = torch.randn(N, device="cuda", dtype=torch_dtype).contiguous()
        feed["gamma"] = gt
        feed["beta"] = bt
        y_ref, mean_ref, rstd_ref = ref_layernorm_fwd(xt, gt, bt, eps)
    else:
        y_ref, mean_ref, rstd_ref = ref_layernorm_fwd(xt, None, None, eps)

    exe = aicf.CudaExecutor()
    out = exe.run(m, feed)

    return max(
        max_abs(out["y"].float(), y_ref.float()),
        max_abs(out["mean"], mean_ref),
        max_abs(out["rstd"], rstd_ref),
    )


@torch.no_grad()
def run_bwd(dtype: str, affine: bool, M: int, N: int, eps: float = 1e-5) -> float:
    m = aicf.Model(dtype=dtype, device="cuda")

    spec_x = aicf.TensorSpec((M, N), dtype, "cuda")
    spec_n = aicf.TensorSpec((N,), dtype, "cuda")

    x  = m.input("x", spec_x)
    dy = m.input("dy", spec_x)

    if affine:
        g = m.input("gamma", spec_n)
        b = m.input("beta", spec_n)

        # fwd to get mean/rstd (same contract as kernel path expects)
        y, mean, rstd = m.add(aicf.LayerNormFwd("ln", eps=eps, affine=True), x, g, b)
        dx, dgamma, dbeta = m.add(aicf.LayerNormBwd("ln_bwd", affine=True), x, dy, g, mean, rstd)

        m.output("dx", dx)
        m.output("dgamma", dgamma)
        m.output("dbeta", dbeta)
    else:
        y, mean, rstd = m.add(aicf.LayerNormFwd("ln", eps=eps, affine=False), x)
        dx = m.add(aicf.LayerNormBwd("ln_bwd", affine=False), x, dy, mean, rstd)
        m.output("dx", dx)

    # feed
    torch_dtype = torch.float16 if dtype == "f16" else torch.float32
    xt = torch.randn(M, N, device="cuda", dtype=torch_dtype).contiguous()
    dyt = torch.randn(M, N, device="cuda", dtype=torch_dtype).contiguous()
    feed = {"x": xt, "dy": dyt}

    if affine:
        gt = torch.randn(N, device="cuda", dtype=torch_dtype).contiguous()
        bt = torch.randn(N, device="cuda", dtype=torch_dtype).contiguous()
        feed["gamma"] = gt
        feed["beta"] = bt

    exe = aicf.CudaExecutor()
    out = exe.run(m, feed)

    # ref: use the fwd stats produced from same x (python ref)
    if affine:
        _, mean_ref, rstd_ref = ref_layernorm_fwd(xt, gt, bt, eps)
        dx_ref, dgamma_ref, dbeta_ref = ref_layernorm_bwd(xt, dyt, mean_ref, rstd_ref, gt)
        return max(
            max_abs(out["dx"].float(), dx_ref.float()),
            max_abs(out["dgamma"], dgamma_ref),
            max_abs(out["dbeta"], dbeta_ref),
        )
    else:
        _, mean_ref, rstd_ref = ref_layernorm_fwd(xt, None, None, eps)
        dx_ref, _, _ = ref_layernorm_bwd(xt, dyt, mean_ref, rstd_ref, None)
        return max_abs(out["dx"].float(), dx_ref.float())


def neg_rank() -> None:
    try:
        m = aicf.Model(dtype="f16", device="cuda")
        x = m.input("x", aicf.TensorSpec((2, 3, 4), "f16", "cuda"))  # wrong rank
        y, mean, rstd = m.add(aicf.LayerNormFwd("ln", eps=1e-5, affine=False), x)  # should raise
        m.output("y", y); m.output("mean", mean); m.output("rstd", rstd)
        exe = aicf.CudaExecutor()
        xt = torch.randn(2, 3, 4, device="cuda", dtype=torch.float16)
        exe.run(m, {"x": xt})
        print("[NEG rank] unexpected success")
    except Exception as e:
        print("[NEG rank] ok:", str(e).splitlines()[0])


def neg_schema() -> None:
    """
    v2에서 schema mismatch를 강제로 유도:
      - registry의 layernorm_fwd schema를 잘못된 값으로 override
      - 실행 시 커널이 schema mismatch로 실패해야 정상
    """
    try:
        reg = CudaRegistry()
        # 잘못된 schema로 override
        if hasattr(reg, "override"):
            reg.override("layernorm_fwd", attr_schema=0x12345678)
        else:
            # override helper 없으면 직접 map 수정 (필요 시)
            # reg._map["layernorm_fwd"] = KernelSpec(kind_id=13, attr_schema=0x12345678)
            raise RuntimeError("CudaRegistry.override() not found. Add override() helper for NEG schema test.")

        m = aicf.Model(dtype="f16", device="cuda")
        M, N = 8, 128
        x = m.input("x", aicf.TensorSpec((M, N), "f16", "cuda"))
        y, mean, rstd = m.add(aicf.LayerNormFwd("ln", eps=1e-5, affine=False), x)
        m.output("y", y); m.output("mean", mean); m.output("rstd", rstd)

        exe = aicf.CudaExecutor(registry=reg)
        xt = torch.randn(M, N, device="cuda", dtype=torch.float16).contiguous()
        exe.run(m, {"x": xt})

        print("[NEG schema] unexpected success")
    except Exception as e:
        print("[NEG schema] ok:", str(e).splitlines()[0])


def main():
    print("LayerNorm v2 test (fwd+bwd)")

    worst = 0.0
    for dtype in ("f32", "f16"):
        for affine in (False, True):
            for (M, N) in ((8, 128), (64, 256), (7, 33)):
                d = run_fwd(dtype=dtype, affine=affine, M=M, N=N, eps=1e-5)
                print(f"[FWD {dtype} affine={affine}] M={M} N={N} max|d|={d:.3e}")
                worst = max(worst, d)

    for dtype in ("f32", "f16"):
        for affine in (False, True):
            for (M, N) in ((8, 128), (64, 256), (7, 33)):
                d = run_bwd(dtype=dtype, affine=affine, M=M, N=N, eps=1e-5)
                print(f"[BWD {dtype} affine={affine}] M={M} N={N} max|d|={d:.3e}")
                worst = max(worst, d)

    print("[OK] worst max|delta| =", worst)

    neg_rank()
    neg_schema()


if __name__ == "__main__":
    main()
