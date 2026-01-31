from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

def fourcc(s: str) -> int:
    assert len(s) == 4
    return int.from_bytes(s.encode("ascii"), "little", signed=False)

@dataclass(frozen=True)
class KernelSpec:
    kind_id: int
    attr_schema: int

class CudaRegistry:
    def __init__(self):
        self._map: Dict[str, KernelSpec] = {
            "add":       KernelSpec(kind_id=0, attr_schema=0),
            "relu":      KernelSpec(kind_id=1, attr_schema=0),
            "gemm":      KernelSpec(kind_id=2, attr_schema=0),
            "bias_add":  KernelSpec(kind_id=3, attr_schema=fourcc("BADD")),

            # optimizer
            "adam_step": KernelSpec(kind_id=10, attr_schema=fourcc("ADAM")),
        }

    def lookup(self, kind: str) -> KernelSpec:
        if kind not in self._map:
            raise KeyError(f"CUDA registry: unknown op kind '{kind}'")
        return self._map[kind]
