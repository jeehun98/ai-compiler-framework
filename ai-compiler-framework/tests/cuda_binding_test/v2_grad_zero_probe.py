# examples/python/python_binding_test/v2_grad_zero_probe.py
from __future__ import annotations
import sys
from pathlib import Path
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]
EX_PY = ROOT / "examples" / "python"
BUILD_PY = ROOT / "build" / "python"
for p in (EX_PY, BUILD_PY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from aicf_cuda import _C


def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def run(dtype: torch.dtype, shape, inplace: bool):
    device = torch.device("cuda:0")

    X = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    if inplace:
        Y = X  # in-place
    else:
        Y = torch.empty(*shape, device=device, dtype=dtype).contiguous()

    # reference: zeros
    Y_ref = torch.zeros_like(X)

    _C.op_call(
        int(_C.OpKind.GradZero),
        [X],
        [Y],
        0,          # schema_id
        b"",        # attrs_bytes
        0,          # stream_handle (current torch stream)
    )

    d = maxabs_delta(Y, Y_ref)
    tag = "inplace" if inplace else "oop"
    print(f"[{dtype}] {tag} shape={tuple(shape)} max|delta|={d:.3e}")
    return d


def main():
    torch.manual_seed(0)

    print("GradZero enum value =", int(_C.OpKind.GradZero))

    # Positive: f32
    worst_f32 = 0.0
    for shape in [(1024,), (64, 256), (8, 32, 128)]:
        worst_f32 = max(worst_f32, run(torch.float32, shape, inplace=False))
    worst_f32 = max(worst_f32, run(torch.float32, (64, 256), inplace=True))
    print("[F32] worst max|delta| =", worst_f32)

    # Positive: f16
    worst_f16 = 0.0
    for shape in [(1024,), (64, 256), (7, 33, 127)]:
        worst_f16 = max(worst_f16, run(torch.float16, shape, inplace=False))
    worst_f16 = max(worst_f16, run(torch.float16, (64, 256), inplace=True))
    print("[F16] worst max|delta| =", worst_f16)

    # NEG: dtype mismatch
    X = torch.randn(128, device="cuda", dtype=torch.float32).contiguous()
    Y_bad = torch.empty(128, device="cuda", dtype=torch.float16).contiguous()
    try:
        _C.op_call(int(_C.OpKind.GradZero), [X], [Y_bad], 0, b"", 0)
        print("[NEG dtype] unexpected OK")
    except RuntimeError as e:
        print("[NEG dtype] ok:", str(e).splitlines()[0])

    # NEG: shape mismatch
    X = torch.randn(128, device="cuda", dtype=torch.float32).contiguous()
    Y_bad = torch.empty(129, device="cuda", dtype=torch.float32).contiguous()
    try:
        _C.op_call(int(_C.OpKind.GradZero), [X], [Y_bad], 0, b"", 0)
        print("[NEG shape] unexpected OK")
    except RuntimeError as e:
        print("[NEG shape] ok:", str(e).splitlines()[0])


if __name__ == "__main__":
    main()
