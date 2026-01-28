from __future__ import annotations
from typing import Sequence
import torch
from aicf_cuda import _C  # your built extension

def current_stream_u64() -> int:
    # torch CUDA stream handle -> int
    s = torch.cuda.current_stream().cuda_stream
    return int(s)

def op_call(kind: int,
            inputs: Sequence[torch.Tensor],
            outputs: Sequence[torch.Tensor],
            attr_schema: int,
            attr_blob: bytes,
            stream: int = 0) -> None:
    _C.op_call(kind, list(inputs), list(outputs), int(attr_schema), attr_blob, int(stream))

def graph_begin() -> int:
    return int(_C.graph_begin())

def graph_end() -> None:
    _C.graph_end()

def graph_launch() -> None:
    _C.graph_launch()

def graph_reset() -> None:
    _C.graph_reset()
