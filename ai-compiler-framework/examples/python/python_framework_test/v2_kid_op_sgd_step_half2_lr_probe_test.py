from __future__ import annotations

import torch

from _test_path_bootstrap import ensure_test_paths, import_cuda_ext

ensure_test_paths()
_C = import_cuda_ext()


def infer_lr(p0: torch.Tensor, g: torch.Tensor, p1: torch.Tensor) -> float:
    # p1 = p0 - lr*g  =>  lr = (p0 - p1) / g  (elementwise)
    # g가 너무 작은 원소는 제외
    eps = 1e-6
    num = (p0 - p1).float()
    den = g.float()
    mask = den.abs() > eps
    if mask.sum().item() == 0:
        return float("nan")
    lr_est = (num[mask] / den[mask]).median().item()
    return float(lr_est)


def main():
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    dtype = torch.float16

    # half2 조건
    N = 256  # 조금 크게 해서 lr 추정 안정화
    p = torch.randn(N, device=device, dtype=dtype).contiguous()
    g = torch.randn(N, device=device, dtype=dtype).contiguous()

    p0 = p.clone()

    kernel_id = "sgd_step_f16_half2_v0"

    # 일단 attrs 없이 실행 (현재 네 테스트와 동일 조건)
    _C.launch_by_id(kernel_id, _C.OpKind.SgdStep, [p, g], [p], 0, b"", 0)

    p1 = p
    lr_est = infer_lr(p0, g, p1)

    # “대충 lr이 반영되긴 했는지”도 같이 확인
    moved = (p0 - p1).abs().float().mean().item()

    print(f"[probe] mean(|delta|)={moved:.6f}, inferred_lr~{lr_est:.8f}")
    print("OK")


if __name__ == "__main__":
    main()
