# python/aicf_v2/src/aicf_v2/emitters/cuda/context.py
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Dict, Any

import _C  # built C++ binding module


def fourcc(s: str) -> int:
    """ASCII 4CC -> little-endian uint32"""
    assert len(s) == 4
    return int.from_bytes(s.encode("ascii"), "little", signed=False)


@dataclass
class CudaEmitContext:
    # --- OpKind Mapping (Based on latest OpKind.h) ---
    EltwiseAdd: int    = int(_C.OpKind.EltwiseAdd)       # 0
    EltwiseRelu: int   = int(_C.OpKind.EltwiseRelu)      # 1
    Gemm: int          = int(_C.OpKind.Gemm)             # 2
    BiasAdd: int       = int(_C.OpKind.BiasAdd)          # 3
    ReduceSum: int     = int(_C.OpKind.ReduceSum)        # 4
    MseGrad: int       = int(_C.OpKind.MseGrad)          # 5
    ReluBwd: int       = int(_C.OpKind.ReluBwd)          # 6
    SgdStep: int       = int(_C.OpKind.SgdStep)          # 7
    Copy: int          = int(_C.OpKind.Copy)             # 8
    GradZero: int      = int(_C.OpKind.GradZero)         # 9
    AdamStep: int      = int(_C.OpKind.AdamStep)         # 10
    StepInc: int       = int(_C.OpKind.StepInc)          # 11
    BiasCorr: int      = int(_C.OpKind.BiasCorr)         # 12
    LayerNormFwd: int  = int(_C.OpKind.LayerNormFwd)     # 13
    LayerNormBwd: int  = int(_C.OpKind.LayerNormBwd)     # 14
    BatchNormFwd: int  = int(_C.OpKind.BatchNormFwd)     # 15
    BatchNormBwd: int  = int(_C.OpKind.BatchNormBwd)     # 16
    GemmEpilogue: int  = int(_C.OpKind.GemmEpilogue)     # 17
    Softmax: int       = int(_C.OpKind.Softmax)          # 18
    SoftmaxBwd: int    = int(_C.OpKind.SoftmaxBwd)       # 19
    MseLoss: int       = int(_C.OpKind.MseLoss)          # 20
    CrossEntropyFwd: int = int(_C.OpKind.CrossEntropyFwd)  # 21
    CrossEntropyBwd: int = int(_C.OpKind.CrossEntropyBwd)  # 22

    # ---- NEW (OpKind.h) ----
    GemmEpilogueBwd: int = int(_C.OpKind.GemmEpilogueBwd)  # 23

    # --- Schema IDs (Lattice ABI) ---
    # NOTE: 값은 C++ 구현(launcher.cu)의 schema_id 상수와 정확히 일치해야 함.
    SCHEMA_ADAM: int = 0x4D414441  # 'ADAM' (LE)
    SCHEMA_BNEP: int = 0x50454E42  # 'BNEP' (LE)
    SCHEMA_LNEP: int = 0x50454E4C  # 'LNEP' (LE)
    SCHEMA_BADD: int = 0x44444142  # 'BADD' (LE)
    SCHEMA_BCOR: int = 0x524F4342  # 'BCOR' (LE)
    SCHEMA_RSUM: int = 0x4D555352  # 'RSUM' (LE)
    SCHEMA_MSEG: int = 0x4745534D  # 'MSEG' (LE)
    SCHEMA_XENT: int = 0x544E4558  # 'XENT' (LE)

    # ---- NEW: GemmEpilogue AttrBlob schema ----
    # C++: static constexpr uint32_t kAttrSchema_GemmEpilogue = 0x4750454Cu; // 'GPEL'
    # IMPORTANT: C++ 상수를 기준으로 "그대로" 맞춘다.
    SCHEMA_GMEP: int = 0x4750454C  # 'GPEL' (matches C++ constant)

    # ---- Dynamic Dispatcher (Python-side) ----
    _op_cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def get_op_module(self, kind: str) -> Any:
        """
        kind 문자열을 기반으로 emitters.cuda 패키지 내의 모듈을 동적으로 가져옵니다.
        - ImportError도 캐시(None)하여 반복 예외 비용을 없앱니다.
        """
        if kind in self._op_cache:
            return self._op_cache[kind]  # None도 캐시됨

        module = None
        try:
            # __name__ = 'aicf_v2.emitters.cuda.context'
            package_path = ".".join(__name__.split(".")[:-1])  # 'aicf_v2.emitters.cuda'
            module = importlib.import_module(f".{kind}", package=package_path)
        except ImportError:
            module = None

        self._op_cache[kind] = module
        return module

    def emit_bwd_for_node(self, b: Any, node: Any, grad_y: int) -> Dict[int, int]:
        fn = getattr(node, "bwd_emit_fn", None)
        if fn is not None:
            return fn(b, self, node, grad_y)

        op_module = self.get_op_module(getattr(node, "kind", ""))
        if op_module is not None and hasattr(op_module, "emit_bwd"):
            return op_module.emit_bwd(b, self, node, grad_y)
        return {}