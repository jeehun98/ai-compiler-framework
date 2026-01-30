from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

def fourcc(s: str) -> int:
    assert len(s) == 4
    return int.from_bytes(s.encode("ascii"), "little")

@dataclass(frozen=True)
class KernelSpec:
    kind_id: int
    attr_schema: int

class CudaRegistry:
    """
    OpKind enum (C++):
      EltwiseAdd  = 0
      EltwiseRelu = 1
      Gemm        = 2
      BiasAdd     = 3
      ...
    """
    def __init__(self):
        # ✅ kind_id는 C++ enum 그대로
        # ✅ schema는 일단 "known good"만 확정, 나머지는 임시로 fourcc로 가정
        self._map: Dict[str, KernelSpec] = {
            "add":      KernelSpec(kind_id=0, attr_schema=fourcc("EADD")),  # 임시(아직 확정 아님)
            "relu":     KernelSpec(kind_id=1, attr_schema=fourcc("EREL")),  # 임시
            "gemm":     KernelSpec(kind_id=2, attr_schema=0),
            "bias_add": KernelSpec(kind_id=3, attr_schema=fourcc("BADD")),  # ✅ 확정(네 테스트 코드)
        }

    def lookup(self, kind: str) -> KernelSpec:
        if kind not in self._map:
            raise KeyError(f"CUDA registry: unknown op kind '{kind}'")
        return self._map[kind]
