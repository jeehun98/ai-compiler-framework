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

            "sgd_step": KernelSpec(kind_id=7, attr_schema=fourcc("SGDS")),

            
            "adam_step": KernelSpec(kind_id=10, attr_schema=fourcc("ADAM")),
            
        
            "layernorm_fwd": KernelSpec(kind_id=13, attr_schema=fourcc("LNEP")),
            "layernorm_bwd": KernelSpec(kind_id=14, attr_schema=0),


            "batchnorm_fwd": KernelSpec(kind_id=15, attr_schema=fourcc("BNEP")),
            "batchnorm_bwd": KernelSpec(kind_id=16, attr_schema=0),
            
        }

    def lookup(self, kind: str) -> KernelSpec:
        if kind not in self._map:
            raise KeyError(f"CUDA registry: unknown op kind '{kind}'")
        return self._map[kind]

    def override(self, kind: str, *, kind_id: int | None = None, attr_schema: int | None = None) -> None:
        ks = self.lookup(kind)
        self._map[kind] = KernelSpec(
            kind_id=ks.kind_id if kind_id is None else int(kind_id),
            attr_schema=ks.attr_schema if attr_schema is None else int(attr_schema),
        )
