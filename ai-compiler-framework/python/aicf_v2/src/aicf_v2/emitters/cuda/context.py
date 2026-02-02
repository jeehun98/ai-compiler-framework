from __future__ import annotations

from dataclasses import dataclass


def fourcc(s: str) -> int:
    assert len(s) == 4
    return int.from_bytes(s.encode("ascii"), "little", signed=False)


@dataclass(frozen=True)
class CudaEmitContext:
    """
    Backend-resolved constants used by emitters.

    - kind_id must match C++ enum OpKind values.
    - schema ids are 0 or fourcc(...) values used by _C.op_call ABI.
    """

    # ---- kind ids (must match your C++ enum OpKind) ----
    EltwiseAdd: int = 0
    EltwiseRelu: int = 1
    Gemm: int = 2
    BiasAdd: int = 3
    ReduceSum: int = 4
    MseGrad: int = 5
    ReluBwd: int = 6
    SgdStep: int = 7
    Copy: int = 8
    GradZero: int = 9
    AdamStep: int = 10
    StepInc: int = 11
    BiasCorr: int = 12
    LayerNormFwd: int = 13
    LayerNormBwd: int = 14
    BatchNormFwd: int = 15
    BatchNormBwd: int = 16
    GemmEpilogue: int = 17

    # ---- schema ids (ABI) ----
    SCHEMA_BADD: int = fourcc("BADD")
    SCHEMA_RSUM: int = fourcc("RSUM")
    SCHEMA_MSEG: int = fourcc("MSEG")
    SCHEMA_SGDS: int = fourcc("SGDS")
    SCHEMA_ADAM: int = fourcc("ADAM")
    SCHEMA_BCOR: int = fourcc("BCOR")
    SCHEMA_LNEP: int = fourcc("LNEP")
    SCHEMA_BNEP: int = fourcc("BNEP")
    SCHEMA_GMEP: int = fourcc("GMEP")
