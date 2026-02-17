from __future__ import annotations
from dataclasses import dataclass
import _C  # 빌드된 C++ 바인딩 모듈

def fourcc(s: str) -> int:
    assert len(s) == 4
    return int.from_bytes(s.encode("ascii"), "little", signed=False)

@dataclass(frozen=True)
class CudaEmitContext:
    """
    Backend-resolved constants used by emitters.
    """

    # ---- kind ids (C++ OpKind Enum과 자동 동기화) ----
    EltwiseAdd: int = int(_C.OpKind.EltwiseAdd)
    EltwiseRelu: int = int(_C.OpKind.EltwiseRelu)
    Gemm: int = int(_C.OpKind.Gemm)
    BiasAdd: int = int(_C.OpKind.BiasAdd)
    ReduceSum: int = int(_C.OpKind.ReduceSum)
    MseGrad: int = int(_C.OpKind.MseGrad)
    ReluBwd: int = int(_C.OpKind.ReluBwd)
    SgdStep: int = int(_C.OpKind.SgdStep)
    Copy: int = int(_C.OpKind.Copy)
    GradZero: int = int(_C.OpKind.GradZero)
    AdamStep: int = int(_C.OpKind.AdamStep)
    StepInc: int = int(_C.OpKind.StepInc)
    BiasCorr: int = int(_C.OpKind.BiasCorr)
    LayerNormFwd: int = int(_C.OpKind.LayerNormFwd)
    LayerNormBwd: int = int(_C.OpKind.LayerNormBwd)
    BatchNormFwd: int = int(_C.OpKind.BatchNormFwd)
    BatchNormBwd: int = int(_C.OpKind.BatchNormBwd)
    GemmEpilogue: int = int(_C.OpKind.GemmEpilogue)
    Softmax: int = int(_C.OpKind.Softmax)
    SoftmaxBwd: int = int(_C.OpKind.SoftmaxBwd)
    MseLoss: int = int(_C.OpKind.MseLoss)
    
    # [신규] CrossEntropy 관련 Kind IDs
    CrossEntropyFwd: int = int(_C.OpKind.CrossEntropyFwd) # KID=21
    CrossEntropyBwd: int = int(_C.OpKind.CrossEntropyBwd) # KID=22

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
    
    # [신규] CrossEntropy Schema ID
    SCHEMA_XENT: int = fourcc("XENT")