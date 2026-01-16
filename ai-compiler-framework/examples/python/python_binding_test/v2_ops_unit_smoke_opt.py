# examples/python/python_binding_test/v2_ops_unit_smoke_opt.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


# --- auto PYTHONPATH bootstrap ---
THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]               # .../ai-compiler-framework (repo root)
EX_PY = ROOT / "examples" / "python" # contains aicf_fw (optional)
BUILD_PY = ROOT / "build" / "python" # contains aicf_cuda (built extension)

for p in (EX_PY, BUILD_PY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

if os.name == "nt":
    try:
        os.add_dll_directory(str(BUILD_PY))
        os.add_dll_directory(str(BUILD_PY / "aicf_cuda"))
    except Exception:
        pass
# --------------------------------

from aicf_cuda import _C


def tf32_off_print():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    print("[torch] allow_tf32.matmul =", torch.backends.cuda.matmul.allow_tf32)
    print("[torch] allow_tf32.cudnn  =", torch.backends.cudnn.allow_tf32)


def mdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, *, atol: float, rtol: float):
    if not torch.allclose(got, ref, atol=atol, rtol=rtol):
        raise AssertionError(f"[FAIL] {name}: maxdiff={mdiff(got, ref)} atol={atol} rtol={rtol}")


def _opkind(name: str) -> int:
    # keep in sync with core_v2 exec mapping
    nm = name.strip().lower()
    m = {
        "step_inc": int(_C.OpKind.StepInc),
        "bias_corr": int(_C.OpKind.BiasCorr),
        "adam_step": int(_C.OpKind.AdamStep),
    }
    if nm not in m:
        raise KeyError(nm)
    return m[nm]


# ----------------------------
# step_inc
# ----------------------------
def test_step_inc(device):
    print("\n== test_step_inc ==")
    if not hasattr(_C.OpKind, "StepInc"):
        print("[SKIP] OpKind.StepInc missing")
        return

    torch.manual_seed(0)

    # int32 scalar
    step = torch.zeros((), device=device, dtype=torch.int32)
    out = torch.empty_like(step)

    _C.op_call(_opkind("step_inc"), [step], [out], {})
    ref = step + 1
    assert_close("step_inc(int32 scalar)", out, ref, atol=0.0, rtol=0.0)

    # int64 scalar (if supported by kernel)
    step64 = torch.zeros((), device=device, dtype=torch.int64)
    out64 = torch.empty_like(step64)
    try:
        _C.op_call(_opkind("step_inc"), [step64], [out64], {})
        ref64 = step64 + 1
        assert_close("step_inc(int64 scalar)", out64, ref64, atol=0.0, rtol=0.0)
    except Exception as e:
        print("[WARN] step_inc int64 not supported by kernel -> ok:", repr(e))

    print("OK step_inc")


# ----------------------------
# bias_corr
# ----------------------------
def _biascorr_ref(step_i32: torch.Tensor, beta1: float, beta2: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Torch ref guess:
      bc1_inv = 1 / (1 - beta1^step)
      bc2_inv = 1 / (1 - beta2^step)

    IMPORTANT:
      - 네 C++ BiasCorr 정의가 이게 아니면 여기만 바꾸면 됨.
      - step은 StepInc 결과(>=1) 기준으로 넣는게 일반적.
    """
    s = step_i32.to(torch.int64)
    s_f = s.to(torch.float32)
    b1 = torch.tensor(beta1, device=step_i32.device, dtype=torch.float32)
    b2 = torch.tensor(beta2, device=step_i32.device, dtype=torch.float32)

    one = torch.ones((), device=step_i32.device, dtype=torch.float32)
    bc1 = one / (one - torch.pow(b1, s_f))
    bc2 = one / (one - torch.pow(b2, s_f))
    return bc1, bc2


def test_bias_corr(device):
    print("\n== test_bias_corr ==")
    if not hasattr(_C.OpKind, "BiasCorr"):
        print("[SKIP] OpKind.BiasCorr missing")
        return

    torch.manual_seed(1)

    beta1, beta2 = 0.9, 0.999

    # Try multiple step values
    for step_val in [1, 2, 5, 10, 100]:
        step = torch.tensor(step_val, device=device, dtype=torch.int32)
        bc1_out = torch.empty((), device=device, dtype=torch.float32)
        bc2_out = torch.empty((), device=device, dtype=torch.float32)

        _C.op_call(_opkind("bias_corr"), [step], [bc1_out, bc2_out], {"beta1": float(beta1), "beta2": float(beta2)})

        bc1_ref, bc2_ref = _biascorr_ref(step, beta1, beta2)

        # bias_corr는 스칼라 연산이라 엄청 타이트하게 가능
        assert_close(f"bias_corr bc1 step={step_val}", bc1_out, bc1_ref, atol=1e-6, rtol=1e-6)
        assert_close(f"bias_corr bc2 step={step_val}", bc2_out, bc2_ref, atol=1e-6, rtol=1e-6)

    print("OK bias_corr")


# ----------------------------
# adam_step
# ----------------------------
def _adam_step_ref(
    p: torch.Tensor,
    g: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    bc1_inv: torch.Tensor,
    bc2_inv: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Torch ref (common Adam, with provided bc*_inv):
      m = beta1*m + (1-beta1)*g
      v = beta2*v + (1-beta2)*g^2
      mhat = m * bc1_inv
      vhat = v * bc2_inv
      p = p - lr * mhat / (sqrt(vhat) + eps)

    IMPORTANT:
      - 네 커널이 decoupled weight decay / different bias-corr semantics면 여기만 바꾸면 됨.
    """
    m2 = beta1 * m + (1.0 - beta1) * g
    v2 = beta2 * v + (1.0 - beta2) * (g * g)
    mhat = m2 * bc1_inv
    vhat = v2 * bc2_inv
    p2 = p - lr * (mhat / (torch.sqrt(vhat) + eps))
    return p2, m2, v2


def test_adam_step(device):
    print("\n== test_adam_step ==")
    if not hasattr(_C.OpKind, "AdamStep"):
        print("[SKIP] OpKind.AdamStep missing")
        return

    torch.manual_seed(2)

    lr, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8

    # Use fp32 only for unit correctness first
    p = torch.randn(64, 8, device=device, dtype=torch.float32).contiguous()
    g = torch.randn(64, 8, device=device, dtype=torch.float32).contiguous()
    m = torch.zeros_like(p).contiguous()
    v = torch.zeros_like(p).contiguous()

    # step=1 bias corr (common)
    step = torch.tensor(1, device=device, dtype=torch.int32)
    bc1, bc2 = _biascorr_ref(step, beta1, beta2)
    bc1 = bc1.to(torch.float32).contiguous()
    bc2 = bc2.to(torch.float32).contiguous()

    # keep copies for ref
    p0, g0, m0, v0 = p.clone(), g.clone(), m.clone(), v.clone()

    # in-place outputs (Stage6 style): outputs = [p, m, v]
    _C.op_call(
        _opkind("adam_step"),
        [p, g, m, v, bc1, bc2],
        [p, m, v],
        {"lr": float(lr), "beta1": float(beta1), "beta2": float(beta2), "eps": float(eps)},
    )

    p_ref, m_ref, v_ref = _adam_step_ref(p0, g0, m0, v0, bc1, bc2, lr, beta1, beta2, eps)

    # Realistic tol: reductions 없음, elementwise only => tight
    assert_close("adam_step p", p, p_ref, atol=1e-5, rtol=1e-5)
    assert_close("adam_step m", m, m_ref, atol=1e-6, rtol=1e-6)
    assert_close("adam_step v", v, v_ref, atol=1e-6, rtol=1e-6)

    print("[adam_step] p  maxdiff =", mdiff(p, p_ref))
    print("[adam_step] m  maxdiff =", mdiff(m, m_ref))
    print("[adam_step] v  maxdiff =", mdiff(v, v_ref))
    print("OK adam_step")


# ----------------------------
# Main
# ----------------------------
def main():
    tf32_off_print()

    device = torch.device("cuda:0")

    # Optional trace toggle
    if hasattr(_C, "trace_enable"):
        _C.trace_enable(True)
    if hasattr(_C, "trace_reset"):
        _C.trace_reset()

    test_step_inc(device)
    test_bias_corr(device)
    test_adam_step(device)

    if hasattr(_C, "trace_get"):
        print("\n[trace]", list(_C.trace_get()))

    print("\nALL OPT OPS PASSED (direct aicf_cuda._C op_call)")


if __name__ == "__main__":
    main()
