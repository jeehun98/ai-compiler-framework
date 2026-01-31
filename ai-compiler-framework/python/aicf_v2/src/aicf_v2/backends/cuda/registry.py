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
            "reduce_sum": KernelSpec(kind_id=4, attr_schema=fourcc("RSUM")),
            "mse_grad":         KernelSpec(kind_id=5, attr_schema=0),
            "mse_grad_scaled":  KernelSpec(kind_id=5, attr_schema=fourcc("MSEG")),
            "relu_bwd": KernelSpec(kind_id=6, attr_schema=0),
            "sgd_step": KernelSpec(kind_id=7, attr_schema=fourcc("SGDS")),
            "copy": KernelSpec(kind_id=8, attr_schema=0),
            "grad_zero": KernelSpec(kind_id=9, attr_schema=0),
            "adam_step": KernelSpec(kind_id=10, attr_schema=fourcc("ADAM")),
            "step_inc": KernelSpec(kind_id=11, attr_schema=0),
            "bias_corr": KernelSpec(kind_id=12, attr_schema=fourcc("BCOR")),
            "layernorm_fwd": KernelSpec(kind_id=13, attr_schema=fourcc("LNEP")),
            "layernorm_bwd": KernelSpec(kind_id=14, attr_schema=0),
            "batchnorm_fwd": KernelSpec(kind_id=15, attr_schema=fourcc("BNEP")),
            "batchnorm_bwd": KernelSpec(kind_id=16, attr_schema=0),
            "gemm_epilogue": KernelSpec(kind_id=17, attr_schema=fourcc("GMEP")),  # schema는 네 C++에 맞춰

            
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
