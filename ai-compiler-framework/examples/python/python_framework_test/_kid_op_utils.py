from __future__ import annotations

import struct
import torch

from _test_path_bootstrap import ensure_test_paths, import_cuda_ext

ensure_test_paths()
_C = import_cuda_ext()


def pack_gemm_attr(transA: bool, transB: bool) -> tuple[int, bytes]:
    # C++: schema_id 0 or 'GEMM' accept. 너 gemm launcher에서 schema_id==0도 parse 하게 해둠.
    # 여긴 명시적으로 GEMM schema로 보냄.
    schema_id = 0x47454D4D  # 'GEMM'
    b = struct.pack("<ii", int(transA), int(transB))
    return schema_id, b


def pack_reduce_sum_attr(axis: int) -> tuple[int, bytes]:
    # C++: schema_id 0 or 'RSUM' accept. reduce_sum launcher에서 axis 읽음.
    schema_id = 0x5253554D  # 'RSUM'
    b = struct.pack("<q", int(axis))  # int64
    return schema_id, b


def launch_by_id(kernel_id: str, kind: "_C.OpKind", inputs: list[torch.Tensor], outputs: list[torch.Tensor],
                 schema_id: int = 0, attrs_bytes: bytes = b"", stream: int = 0):
    # pybind sig:
    # (kernel_id: str, kind: aicf_cuda._C.OpKind, inputs: Sequence, outputs: Sequence, schema_id: int, attrs_bytes: bytes, stream: int)
    _C.launch_by_id(str(kernel_id), kind, inputs, outputs, int(schema_id), attrs_bytes, int(stream))


def assert_allclose(a: torch.Tensor, b: torch.Tensor, atol=1e-2, rtol=1e-2, msg=""):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"allclose failed: max_abs_diff={diff} {msg}")
