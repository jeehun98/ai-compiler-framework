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

def schema_4cc(name: str) -> int:
    assert len(name) == 4
    b = name.encode("ascii")
    return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)

SCHEMA_BNEP = schema_4cc("BNEP")

def make_attr_bnep(eps: float, use_running_stats: bool) -> tuple[int, bytes]:
    flags = 1 if use_running_stats else 0
    return SCHEMA_BNEP, struct.pack("<fI", float(eps), int(flags))

def max_abs(a, b):
    return (a - b).abs().max().item()

@torch.no_grad()
def run_fwd_training(N=8, C=16, H=8, W=8, affine=True, eps=1e-5):
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)
    save_mean = torch.zeros(C, device="cuda", dtype=torch.float32)
    save_rstd = torch.zeros(C, device="cuda", dtype=torch.float32)

    if affine:
        gamma = torch.randn(C, device="cuda", dtype=torch.float16)
        beta  = torch.randn(C, device="cuda", dtype=torch.float16)
        inputs = [x, gamma, beta]
    else:
        inputs = [x]

    outputs = [y, save_mean, save_rstd]
    schema_id, attrs = make_attr_bnep(eps, use_running_stats=False)

    _C.op_call(_C.OpKind.BatchNormFwd, inputs, outputs, schema_id, attrs)

    # torch ref (training stats)
    if affine:
        y_ref = torch.nn.functional.batch_norm(
            x, running_mean=None, running_var=None,
            weight=gamma, bias=beta,
            training=True, momentum=0.0, eps=eps
        )
    else:
        y_ref = torch.nn.functional.batch_norm(
            x, running_mean=None, running_var=None,
            weight=None, bias=None,
            training=True, momentum=0.0, eps=eps
        )

    d = max_abs(y.float(), y_ref.float())
    return d

@torch.no_grad()
def run_fwd_infer(N=8, C=16, H=8, W=8, affine=True, eps=1e-5):
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)

    running_mean = torch.randn(C, device="cuda", dtype=torch.float32)
    running_var  = torch.rand(C, device="cuda", dtype=torch.float32) + 0.5  # positive

    if affine:
        gamma = torch.randn(C, device="cuda", dtype=torch.float16)
        beta  = torch.randn(C, device="cuda", dtype=torch.float16)
        inputs = [x, gamma, beta, running_mean, running_var]
    else:
        inputs = [x, running_mean, running_var]

    outputs = [y]
    schema_id, attrs = make_attr_bnep(eps, use_running_stats=True)

    _C.op_call(_C.OpKind.BatchNormFwd, inputs, outputs, schema_id, attrs)

    y_ref = torch.nn.functional.batch_norm(
        x, running_mean=running_mean, running_var=running_var,
        weight=(gamma if affine else None),
        bias=(beta if affine else None),
        training=False, momentum=0.0, eps=eps
    )

    d = max_abs(y.float(), y_ref.float())
    return d

def run_bwd_training(N=8, C=16, H=8, W=8, eps=1e-5):
    # inputs (ours) - no_grad로 해도 됨
    with torch.no_grad():
        x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
        gamma = torch.randn(C, device="cuda", dtype=torch.float16)
        beta  = torch.randn(C, device="cuda", dtype=torch.float16)

        # our forward to get save_mean/save_rstd
        y = torch.empty_like(x)
        save_mean = torch.zeros(C, device="cuda", dtype=torch.float32)
        save_rstd = torch.zeros(C, device="cuda", dtype=torch.float32)

        schema_id, attrs = make_attr_bnep(eps, use_running_stats=False)
        _C.op_call(_C.OpKind.BatchNormFwd, [x, gamma, beta], [y, save_mean, save_rstd], schema_id, attrs)

        dy = torch.randn_like(x)

        dx = torch.empty_like(x)
        dgamma = torch.zeros(C, device="cuda", dtype=torch.float32)
        dbeta  = torch.zeros(C, device="cuda", dtype=torch.float32)

        _C.op_call(_C.OpKind.BatchNormBwd, [x, dy, gamma, save_mean, save_rstd], [dx, dgamma, dbeta])

    # torch ref gradients (여긴 no_grad 절대 걸면 안됨)
    x2 = x.detach().clone().requires_grad_(True)
    g2 = gamma.detach().clone().requires_grad_(True)
    b2 = beta.detach().clone().requires_grad_(True)

    y_ref = torch.nn.functional.batch_norm(
        x2, running_mean=None, running_var=None,
        weight=g2, bias=b2,
        training=True, momentum=0.0, eps=eps
    )
    y_ref.backward(dy)  # dy는 requires_grad 없어도 됨

    d_dx = max_abs(dx.float(), x2.grad.float())
    d_dg = max_abs(dgamma, g2.grad.float())
    d_db = max_abs(dbeta,  b2.grad.float())
    return max(d_dx, d_dg, d_db)

def main():
    print(f"BatchNormFwd enum value = {int(_C.OpKind.BatchNormFwd)}")
    print(f"BatchNormBwd enum value = {int(_C.OpKind.BatchNormBwd)}")
    print(f"schema_id(BNEP) = 0x{SCHEMA_BNEP:08x}")

    worst = 0.0

    for (N,C,H,W) in [(8,16,8,8), (16,32,16,16), (7,33,5,7)]:
        d = run_fwd_training(N,C,H,W, affine=True, eps=1e-5)
        print(f"[FWD train affine] N={N} C={C} H={H} W={W} max|d|={d:.3e}")
        worst = max(worst, d)

        d = run_fwd_training(N,C,H,W, affine=False, eps=1e-5)
        print(f"[FWD train noaff ] N={N} C={C} H={H} W={W} max|d|={d:.3e}")
        worst = max(worst, d)

        d = run_fwd_infer(N,C,H,W, affine=True, eps=1e-5)
        print(f"[FWD infer affine] N={N} C={C} H={H} W={W} max|d|={d:.3e}")
        worst = max(worst, d)

        d = run_fwd_infer(N,C,H,W, affine=False, eps=1e-5)
        print(f"[FWD infer noaff ] N={N} C={C} H={H} W={W} max|d|={d:.3e}")
        worst = max(worst, d)

        d = run_bwd_training(N,C,H,W, eps=1e-5)
        print(f"[BWD train affine] N={N} C={C} H={H} W={W} max|d|={d:.3e}")
        worst = max(worst, d)

    print(f"[OK] worst max|delta| = {worst}")

    # NEG: wrong rank
    try:
        x = torch.randn(8, 16, device="cuda", dtype=torch.float16)
        y = torch.empty_like(x)
        schema_id, attrs = make_attr_bnep(1e-5, use_running_stats=True)
        _C.op_call(_C.OpKind.BatchNormFwd, [x], [y], schema_id, attrs)
        print("[NEG rank] unexpected success")
    except RuntimeError as e:
        print(f"[NEG rank] ok: {e}")

    # NEG: wrong schema
    try:
        x = torch.randn(8,16,8,8, device="cuda", dtype=torch.float16)
        y = torch.empty_like(x)
        rm = torch.zeros(16, device="cuda", dtype=torch.float32)
        rv = torch.ones(16, device="cuda", dtype=torch.float32)
        _C.op_call(_C.OpKind.BatchNormFwd, [x, rm, rv], [y], 0x12345678, b"\x00"*8)
        print("[NEG schema] unexpected success")
    except RuntimeError as e:
        print(f"[NEG schema] ok: {e}")

if __name__ == "__main__":
    main()
