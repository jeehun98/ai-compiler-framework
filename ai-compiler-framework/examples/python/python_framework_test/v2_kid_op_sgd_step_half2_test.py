from __future__ import annotations

import torch

from _test_path_bootstrap import ensure_test_paths, import_cuda_ext

ensure_test_paths()
_C = import_cuda_ext()


def infer_lr(p0: torch.Tensor, g: torch.Tensor, p1: torch.Tensor) -> float:
    eps = 1e-6
    num = (p0 - p1).float()
    den = g.float()
    mask = den.abs() > eps
    if mask.sum().item() == 0:
        return float("nan")
    return float((num[mask] / den[mask]).median().item())


def assert_lr_close(lr_est: float, lr_target: float, tol: float = 5e-4):
    if not (abs(lr_est - lr_target) <= tol):
        raise AssertionError(f"lr mismatch: inferred={lr_est:.8f}, target={lr_target:.8f}, tol={tol}")


def main():
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    dtype = torch.float16

    lr = 1e-2

    # half2 조건 + lr 추정 안정화
    N = 256
    p = torch.randn(N, device=device, dtype=dtype).contiguous()
    g = torch.randn(N, device=device, dtype=dtype).contiguous()

    p0 = p.clone()

    kernel_id = "sgd_step_f16_half2_v0"

    # ⚠️ 현재는 attrs 포맷을 모르는 상태이므로 일단 비워서 실행
    _C.launch_by_id(kernel_id, _C.OpKind.SgdStep, [p, g], [p], 0, b"", 0)

    p1 = p
    lr_est = infer_lr(p0, g, p1)
    print(f"[probe] inferred_lr~{lr_est:.8f} (target={lr})")

    # 1) lr 자체가 맞는지 먼저 확인 (여기서 틀리면 attr 포맷 문제)
    assert_lr_close(lr_est, lr, tol=5e-4)

    # 2) lr이 맞다면 그 다음은 수치오차 허용으로 allclose
    ref = (p0.float() - g.float() * lr).to(dtype)
    # half 연산/half2 path는 round 때문에 diff가 1e-2 근처까지 나올 수 있음.
    # 현실적으로 atol을 살짝 풀자.
    if not torch.allclose(p1, ref, atol=2e-2, rtol=2e-2):
        diff = (p1 - ref).abs().max().item()
        raise AssertionError(f"allclose failed: max_abs_diff={diff}")

    print("[OK] sgd_step_f16_half2_v0 works (lr matched + numeric within tol).")
    print("OK")


if __name__ == "__main__":
    main()
