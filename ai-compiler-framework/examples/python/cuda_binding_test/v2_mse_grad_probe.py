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

from aicf_cuda import _C


def pack_mse_grad_scale(scale: float) -> bytes:
    # schema_id는 op_call의 schema_id로 주고, attrs_bytes는 payload만 주는 형태면:
    # 지금 네 v2 ABI는 (schema_id, bytes, data)를 C++로 전달한다고 했지.
    # 하지만 현재 op_call 시그니처가 (schema_id, attrs_bytes)라서,
    # C++ 쪽에서 AttrBlob을 만들어 전달하는 구현일 가능성이 큼.
    #
    # 너 gemm/bias_add가 "schema_id=0 + attrs_bytes"로 동작하고 있으니,
    # 여기서도 동일하게: schema_id=0, attrs_bytes=payload 로 가정.
    return struct.pack("<f", float(scale))


def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def ref_mse_grad(pred: torch.Tensor, target: torch.Tensor, scale: float) -> torch.Tensor:
    return (pred - target) * scale


def run_f32(shape, scale: float | None):
    pred = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    targ = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    g = torch.empty_like(pred).contiguous()

    numel = pred.numel()
    s = (2.0 / float(numel)) if scale is None else float(scale)

    ref = ref_mse_grad(pred, targ, s)

    schema_id = 0
    attrs = b""
    if scale is not None:
        # C++는 schema_id=='MSEG'일 때만 scale을 읽도록 했으니,
        # 여기서는 schema_id를 'MSEG'로 전달해야 함.
        schema_id = 0x4745534D  # 'MSEG'
        attrs = pack_mse_grad_scale(s)

    _C.op_call(int(_C.OpKind.MseGrad), [pred, targ], [g], schema_id, attrs, 0)

    d = maxabs_delta(g, ref)
    tag = "default" if scale is None else f"scale={s:g}"
    print(f"[F32] shape={tuple(shape)} {tag} max|delta|={d:.3e}")
    return d


def run_f16(shape, scale: float | None):
    pred = torch.randn(*shape, device="cuda", dtype=torch.float16).contiguous()
    targ = torch.randn(*shape, device="cuda", dtype=torch.float16).contiguous()
    g = torch.empty_like(pred).contiguous()

    numel = pred.numel()
    s = (2.0 / float(numel)) if scale is None else float(scale)

    ref = ref_mse_grad(pred.float(), targ.float(), s).half()

    schema_id = 0
    attrs = b""
    if scale is not None:
        schema_id = 0x4745534D  # 'MSEG'
        attrs = pack_mse_grad_scale(s)

    _C.op_call(int(_C.OpKind.MseGrad), [pred, targ], [g], schema_id, attrs, 0)

    d = maxabs_delta(g, ref)
    tag = "default" if scale is None else f"scale={s:g}"
    print(f"[F16] shape={tuple(shape)} {tag} max|delta|={d:.3e}")
    return d


def main():
    torch.manual_seed(0)

    print("MseGrad enum value =", int(_C.OpKind.MseGrad))
    print("schema_id(MSEG) =", hex(0x4745534D))

    # Positive: default scale
    worst = 0.0
    worst = max(worst, run_f32((64, 256), None))
    worst = max(worst, run_f32((8, 32, 128), None))
    print("[F32] worst max|delta| =", worst)

    worst16 = 0.0
    worst16 = max(worst16, run_f16((64, 256), None))
    worst16 = max(worst16, run_f16((7, 33, 127), None))
    print("[F16] worst max|delta| =", worst16)

    # Positive: explicit scale (schema MSEG)
    run_f32((64, 256), 0.125)
    run_f16((64, 256), 0.125)

    # NEG: shape mismatch
    p = torch.randn(8, 7, device="cuda", dtype=torch.float32).contiguous()
    t = torch.randn(8, 8, device="cuda", dtype=torch.float32).contiguous()
    g = torch.empty_like(p).contiguous()
    try:
        _C.op_call(int(_C.OpKind.MseGrad), [p, t], [g], 0, b"", 0)
        print("[NEG shape] unexpected OK")
    except RuntimeError as e:
        print("[NEG shape] ok:", str(e).splitlines()[0])

    # NEG: dtype mismatch
    p = torch.randn(8, 7, device="cuda", dtype=torch.float32).contiguous()
    t = torch.randn(8, 7, device="cuda", dtype=torch.float16).contiguous()
    g = torch.empty_like(p).contiguous()
    try:
        _C.op_call(int(_C.OpKind.MseGrad), [p, t], [g], 0, b"", 0)
        print("[NEG dtype] unexpected OK")
    except RuntimeError as e:
        print("[NEG dtype] ok:", str(e).splitlines()[0])


if __name__ == "__main__":
    main()
