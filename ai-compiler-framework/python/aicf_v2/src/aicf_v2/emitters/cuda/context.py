from __future__ import annotations
import importlib
from dataclasses import dataclass, field
from typing import Dict, Any, List
import _C  # 빌드된 C++ 바인딩 모듈

def fourcc(s: str) -> int:
    assert len(s) == 4
    return int.from_bytes(s.encode("ascii"), "little", signed=False)

@dataclass
class CudaEmitContext:
    # --- OpKind Mapping (Based on latest OpKind.h) ---
    EltwiseAdd: int    = int(_C.OpKind.EltwiseAdd)    # 0
    EltwiseRelu: int   = int(_C.OpKind.EltwiseRelu)   # 1
    Gemm: int          = int(_C.OpKind.Gemm)          # 2
    BiasAdd: int       = int(_C.OpKind.BiasAdd)       # 3
    ReduceSum: int     = int(_C.OpKind.ReduceSum)     # 4
    MseGrad: int       = int(_C.OpKind.MseGrad)       # 5
    ReluBwd: int       = int(_C.OpKind.ReluBwd)       # 6
    SgdStep: int       = int(_C.OpKind.SgdStep)       # 7
    Copy: int          = int(_C.OpKind.Copy)          # 8
    GradZero: int      = int(_C.OpKind.GradZero)      # 9
    AdamStep: int      = int(_C.OpKind.AdamStep)      # 10
    StepInc: int       = int(_C.OpKind.StepInc)       # 11
    BiasCorr: int      = int(_C.OpKind.BiasCorr)      # 12
    LayerNormFwd: int  = int(_C.OpKind.LayerNormFwd)  # 13
    LayerNormBwd: int  = int(_C.OpKind.LayerNormBwd)  # 14
    BatchNormFwd: int  = int(_C.OpKind.BatchNormFwd)  # 15
    BatchNormBwd: int  = int(_C.OpKind.BatchNormBwd)  # 16
    GemmEpilogue: int  = int(_C.OpKind.GemmEpilogue)  # 17
    Softmax: int       = int(_C.OpKind.Softmax)       # 18
    SoftmaxBwd: int    = int(_C.OpKind.SoftmaxBwd)    # 19
    MseLoss: int       = int(_C.OpKind.MseLoss)       # 20
    CrossEntropyFwd: int = int(_C.OpKind.CrossEntropyFwd) # 21
    CrossEntropyBwd: int = int(_C.OpKind.CrossEntropyBwd) # 22

    # --- Schema IDs (Lattice ABI 규격) ---
    # 프로젝트에서 약속된 4바이트 Magic Number들입니다.
    SCHEMA_ADAM: int = 0x4144414D  # 'ADAM'
    SCHEMA_BNEP: int = 0x424E4550  # 'BNEP'
    SCHEMA_LNEP: int = 0x4C4E4550  # 'LNEP'
    SCHEMA_BADD: int = 0x42414444  # 'BADD'
    SCHEMA_BCOR: int = 0x42434F52  # 'BCOR'
    SCHEMA_RSUM: int = 0x5253554D  # 'RSUM'
    SCHEMA_MSEG: int = 0x4D534547  # 'MSEG'
    SCHEMA_XENT: int = 0x58454E54  # 'XENT'

    # ---- 3. Dynamic Dispatcher (Mirroring BWD Core) ----
    # 로드된 모듈을 캐싱하여 성능 최적화
    _op_cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def get_op_module(self, kind: str) -> Any:
        """kind 문자열을 기반으로 emitters.cuda 패키지 내의 모듈을 동적으로 가져옵니다."""
        if kind in self._op_cache:
            return self._op_cache[kind]

        try:
            # 규격화된 파일 구조 (예: .gemm, .batchnorm)에서 모듈 로드
            # __name__은 'aicf_v2.emitters.cuda.context' 이므로 상위 패키지에서 검색
            package_path = ".".join(__name__.split(".")[:-1])
            module = importlib.import_module(f".{kind}", package=package_path)
            self._op_cache[kind] = module
            return module
        except ImportError:
            # 최적화된 이름(fused_*)이나 미분 규칙이 없는 연산에 대한 처리
            return None

    def emit_bwd_for_node(self, b: Builder, node: Any, grad_y: int) -> Dict[int, int]:
        """
        FWD 노드의 정보를 바탕으로 대응하는 BWD 연산을 Builder에 추가 누적합니다.
        기록 누적 방식의 Autograd 핵심 인터페이스입니다.
        """
        op_module = self.get_op_module(node.kind)
        
        if op_module and hasattr(op_module, "emit_bwd"):
            # 공통 규격: b, ctx, fwd_node, grad_y를 인자로 전달
            # 반환값: Dict[input_vid, grad_vid]
            return op_module.emit_bwd(b, self, node, grad_y)
        
        # 미분 규칙이 없는 경우(Optimizer Step 등) 빈 결과 반환
        return {}