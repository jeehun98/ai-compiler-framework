from __future__ import annotations

import torch

from _test_path_bootstrap import ensure_test_paths, import_cuda_ext
from _kid_op_utils import pack_reduce_sum_attr, launch_by_id, assert_allclose

ensure_test_paths()
_C = import_cuda_ext()


def main():
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    # dY: [M,N] (contig), out dB: [N] float32 contig
    M, N = 64, 8
    dY = torch.randn(M, N, device=device, dtype=torch.float16).contiguous()
    dB = torch.empty(N, device=device, dtype=torch.float32).contiguous()

    # axis=0 => sum over rows => shape [N]
    ref = dY.float().sum(dim=0)

    kernel_id = "reduce_sum_lastdim_f16_to_f32_v0"
    schema_id, attrs = pack_reduce_sum_attr(axis=0)

    launch_by_id(kernel_id, _C.OpKind.ReduceSum, [dY], [dB], schema_id, attrs, stream=0)

    assert_allclose(dB, ref, atol=1e-3, rtol=1e-3, msg="ReduceSum f16->f32 mismatch")
    print("[OK] reduce_sum_lastdim_f16_to_f32_v0 axis=0 works.")

    # 실패 케이스도 하나 박아두면 좋음: axis!=0면 InvalidArgument가 정상
    # (지금 launcher가 axis!=0에서 InvalidArgument 리턴하게 돼있음)
    bad = torch.empty_like(dB)
    schema_id2, attrs2 = pack_reduce_sum_attr(axis=1)
    try:
        launch_by_id(kernel_id, _C.OpKind.ReduceSum, [dY], [bad], schema_id2, attrs2, stream=0)
        raise RuntimeError("Expected failure for axis=1, but succeeded.")
    except RuntimeError as e:
        print("[OK] axis=1 correctly fails:", str(e).splitlines()[0])

    print("OK")


if __name__ == "__main__":
    main()
