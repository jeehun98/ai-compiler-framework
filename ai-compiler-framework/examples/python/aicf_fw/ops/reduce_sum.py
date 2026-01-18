from __future__ import annotations
import torch
from ..backend import cuda, abi
from ..backend.validate import ensure_cuda, ensure_contig

# reduce_sum_keep_lastdim: dB[j] = sum_i dY[i,j], axis fixed 0 in backend
def reduce_sum(dY: torch.Tensor, dB: torch.Tensor, axis: int = 0, stream: int = 0) -> None:
    ensure_cuda(dY, "dY"); ensure_cuda(dB, "dB")
    ensure_contig(dY, "dY"); ensure_contig(dB, "dB")
    blob = abi.pack_reduce_sum(axis=axis)
    cuda.op_call(kind=int(cuda._C.OpKind.ReduceSum),  # or your own enum mapping
                inputs=[dY],
                outputs=[dB],
                attr_schema=abi.AttrSchema.REDUCE_SUM,
                attr_blob=blob,
                stream=stream)
