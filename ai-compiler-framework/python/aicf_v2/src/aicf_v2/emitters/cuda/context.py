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
    
    - kind_id는 이제 _C.OpKind를 직접 참조하여 C++ Enum과 항상 일치합니다.
    - schema ids는 _C.op_call ABI에서 사용하는 속성 레이아웃 식별자입니다.
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
    Softmax: int = int(_C.OpKind.Softmax)        # 오타 수정 완료
    SoftmaxBwd: int = int(_C.OpKind.SoftmaxBwd)

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