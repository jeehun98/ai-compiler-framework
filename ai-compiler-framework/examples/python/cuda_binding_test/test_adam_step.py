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


def check_close(name, got, ref, atol=1e-5, rtol=1e-5, fatal=True):
    max_abs = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[{name}] ok={ok} max_abs={max_abs} atol={atol} rtol={rtol}")
    if (not ok) and fatal:
        # print a tiny slice for quick diagnosis
        d = (got - ref).detach()
        print("  slice got    :", got[:5].detach().cpu())
        print("  slice ref    :", ref[:5].detach().cpu())
        print("  slice diff   :", d[:5].detach().cpu())
        raise RuntimeError(f"{name} mismatch")
    return ok, max_abs


@torch.inference_mode()
def aicf_adam_step_(p, g, m, v, *, lr, beta1, beta2, eps, t):
    # bias-correction scalars computed on host (match kernel contract)
    bc1_inv = 1.0 / (1.0 - (beta1 ** t))
    bc2_inv = 1.0 / (1.0 - (beta2 ** t))
    attrs = {
        "lr": float(lr),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "eps": float(eps),
        "bc1_inv": float(bc1_inv),
        "bc2_inv": float(bc2_inv),
    }
    op_call(aicf.OpKind.AdamStep, [p, g, m, v], [p, m, v], attrs)


@torch.inference_mode()
def test_adam_step_matches_torch_single_step_report_only():
    """
    참고용: torch.optim.Adam과 1-step 비교.
    PyTorch 내부 옵션/경로 차이로 완전일치가 깨질 수 있어서
    FAIL이어도 테스트를 죽이지 않고 리포트만 출력한다.
    """
    torch.manual_seed(0)

    n = 4096
    lr = 1e-3
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    t = 1

    # --- prepare tensors ---
    p0 = torch.randn(n, device="cuda", dtype=torch.float32).contiguous()
    g = torch.randn_like(p0).contiguous()

    # --- torch Adam reference ---
    p_ref = p0.clone().detach().requires_grad_(True)
    p_ref.grad = g.clone()

    # try to lock down options (some args may not exist in your torch version)
    opt_kwargs = dict(lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=0.0, amsgrad=False)
    try:
        opt = torch.optim.Adam([p_ref], **opt_kwargs, foreach=False, fused=False)
    except TypeError:
        opt = torch.optim.Adam([p_ref], **opt_kwargs)

    opt.step()
    torch.cuda.synchronize()

    # --- AICF op ---
    p = p0.clone()
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    aicf_adam_step_(p, g, m, v, lr=lr, beta1=beta1, beta2=beta2, eps=eps, t=t)
    torch.cuda.synchronize()

    # report only
    ok, max_abs = check_close(
        "AdamStep vs torch Adam (t=1) [report-only]",
        p, p_ref.detach(),
        atol=2e-5, rtol=2e-5,
        fatal=False
    )

    if not ok:
        # extra diagnostics: magnitude check (first step often moves ~lr * sign(g))
        diff = (p - p_ref.detach()).detach()
        print("  mean_abs_diff:", diff.abs().mean().item())
        print("  max_abs_diff :", diff.abs().max().item())
        # print a small slice
        print("  p[:5]        :", p[:5].detach().cpu())
        print("  p_ref[:5]    :", p_ref.detach()[:5].detach().cpu())
        print("  g[:5]        :", g[:5].detach().cpu())


@torch.inference_mode()
def test_adam_step_multi_step_matches_manual(steps=5):
    """
    메인 PASS 기준: 커널 수식과 동일한 manual reference(같은 dtype/디바이스)로 multi-step 비교.
    이게 통과하면 adam_step 수식/구현은 맞다고 보면 됨.
    """
    torch.manual_seed(1)

    n = 8192
    lr = 3e-4
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    p = torch.randn(n, device="cuda", dtype=torch.float32).contiguous()
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)

    # reference copies
    p_ref = p.clone()
    m_ref = m.clone()
    v_ref = v.clone()

    for t in range(1, steps + 1):
        g = torch.randn_like(p).contiguous()

        # AICF op
        aicf_adam_step_(p, g, m, v, lr=lr, beta1=beta1, beta2=beta2, eps=eps, t=t)

        # manual ref on GPU (same math as kernel)
        m_ref.mul_(beta1).add_(g, alpha=(1.0 - beta1))
        v_ref.mul_(beta2).addcmul_(g, g, value=(1.0 - beta2))

        bc1_inv = 1.0 / (1.0 - (beta1 ** t))
        bc2_inv = 1.0 / (1.0 - (beta2 ** t))

        denom = (v_ref * bc2_inv).sqrt().add(eps)
        p_ref.addcdiv_(m_ref * bc1_inv, denom, value=-lr)

    torch.cuda.synchronize()
    check_close(
        f"AdamStep multi-step manual ({steps})",
        p, p_ref,
        atol=2e-5, rtol=2e-5,
        fatal=True
    )


@torch.inference_mode()
def test_adam_step_cudagraph_capture_safe(iters=20, replays=200):
    """
    PyTorch CUDA Graph capture/replay safety.
    Capture must happen on a non-default stream.
    """
    n = 1_000_000
    lr = 1e-3
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    # fixed tensors during capture (stable addresses)
    p = torch.randn(n, device="cuda", dtype=torch.float32).contiguous()
    g = torch.randn_like(p).contiguous()
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)

    # warmup
    for t in range(1, 3):
        aicf_adam_step_(p, g, m, v, lr=lr, beta1=beta1, beta2=beta2, eps=eps, t=t)
    torch.cuda.synchronize()

    # capture on non-default stream
    s = torch.cuda.Stream()
    cg = torch.cuda.CUDAGraph()

    # static attrs during capture (graph is static)
    t_cap = 1
    bc1_inv = 1.0 / (1.0 - (beta1 ** t_cap))
    bc2_inv = 1.0 / (1.0 - (beta2 ** t_cap))
    attrs = {
        "lr": float(lr),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "eps": float(eps),
        "bc1_inv": float(bc1_inv),
        "bc2_inv": float(bc2_inv),
    }

    torch.cuda.synchronize()
    with torch.cuda.stream(s):
        cg.capture_begin()
        for _ in range(iters):
            op_call(aicf.OpKind.AdamStep, [p, g, m, v], [p, m, v], attrs)
        cg.capture_end()
    torch.cuda.synchronize()

    # perturb and replay
    p.copy_(torch.randn_like(p))
    m.zero_()
    v.zero_()
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::adam_step_cudagraph_replay"):
        for _ in range(replays):
            cg.replay()
    torch.cuda.synchronize()

    print("[AdamStep(CUDAGraph)] replay OK (no crash)")


@torch.inference_mode()
def test_adam_step_aicf_capture_safe(iters=20, replays=200):
    """
    AICF capture/replay safety using your dedicated stream policy.
    """
    n = 1_000_000
    lr = 1e-3
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    p = torch.randn(n, device="cuda", dtype=torch.float32).contiguous()
    g = torch.randn_like(p).contiguous()
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)

    # warmup
    for t in range(1, 3):
        aicf_adam_step_(p, g, m, v, lr=lr, beta1=beta1, beta2=beta2, eps=eps, t=t)
    torch.cuda.synchronize()

    # static attrs during capture
    t_cap = 1
    bc1_inv = 1.0 / (1.0 - (beta1 ** t_cap))
    bc2_inv = 1.0 / (1.0 - (beta2 ** t_cap))
    attrs = {
        "lr": float(lr),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "eps": float(eps),
        "bc1_inv": float(bc1_inv),
        "bc2_inv": float(bc2_inv),
    }

    aicf.capture_begin()
    for _ in range(iters):
        op_call(aicf.OpKind.AdamStep, [p, g, m, v], [p, m, v], attrs)
    aicf.capture_end()
    torch.cuda.synchronize()

    # perturb and replay
    p.copy_(torch.randn_like(p))
    m.zero_()
    v.zero_()
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::adam_step_aicf_replay"):
        for _ in range(replays):
            aicf.replay()
    torch.cuda.synchronize()

    print("[AdamStep(AICF Graph)] replay OK (no crash)")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.cuda.synchronize()

    # 참고용 리포트 (FAIL이어도 통과 처리)
    test_adam_step_matches_torch_single_step_report_only()

    # 메인 PASS 기준
    test_adam_step_multi_step_matches_manual(steps=5)

    # capture-safe
    test_adam_step_cudagraph_capture_safe()
    test_adam_step_aicf_capture_safe()

    print("ALL OK")

# Example ncu:
# ncu --target-processes all --launch-count 1 --section SpeedOfLight --page details \
#   -o adam_step_detail python .\test_adam_step.py
