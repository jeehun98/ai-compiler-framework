from __future__ import annotations
import struct

class AttrSchema:
    NONE = 0
    REDUCE_SUM = 1
    GEMM = 2
    # add more...

def pack_none() -> bytes:
    return b""

def pack_reduce_sum(axis: int = 0) -> bytes:
    # int64 axis
    return struct.pack("<q", int(axis))

def pack_gemm(trans_a: int = 0, trans_b: int = 0) -> bytes:
    # two int32
    return struct.pack("<ii", int(trans_a), int(trans_b))
