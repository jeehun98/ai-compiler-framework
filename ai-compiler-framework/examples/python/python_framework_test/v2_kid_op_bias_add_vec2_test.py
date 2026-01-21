from __future__ import annotations

import torch

from _test_path_bootstrap import ensure_test_paths, import_cuda_ext
from _kid_op_utils import launch_by_id, assert_allclose

ensure_test_paths()
_C = import_cuda_ext()


def main():
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    dtype = torch.float16

    # vec2 조건: lastdim even + 4B align 만족시키는 shape
    B, D = 64, 8
    Y = torch.randn(B, D, device=device, dtype=dtype).contiguous()
    b = torch.randn(D, device=device, dtype=dtype).contiguous()

    out = Y.clone()

    ref = Y + b.view(1, D)

    kernel_id = "bias_add_f16_vec2_v0"
    launch_by_id(kernel_id, _C.OpKind.BiasAdd, [out, b], [out], schema_id=0, attrs_bytes=b"", stream=0)

    assert_allclose(out, ref, atol=1e-2, rtol=1e-2, msg="bias_add vec2 mismatch")
    print("[OK] bias_add_f16_vec2_v0 works.")
    print("OK")


if __name__ == "__main__":
    main()
