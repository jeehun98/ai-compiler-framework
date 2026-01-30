from __future__ import annotations
from typing import Sequence
import torch

_C_mod = None

def _get_C():
    global _C_mod
    if _C_mod is None:
        from aicf_v2 import _C as _C_loaded
        _C_mod = _C_loaded
    return _C_mod

def current_stream_u64() -> int:
    s = torch.cuda.current_stream().cuda_stream
    return int(s)

def op_call(kind_id: int,
            inputs: Sequence[torch.Tensor],
            outputs: Sequence[torch.Tensor],
            attr_schema: int,
            attr_blob: bytes,
            stream: int = 0) -> None:
    C = _get_C()
    C.op_call(int(kind_id), list(inputs), list(outputs), int(attr_schema), attr_blob, int(stream))
