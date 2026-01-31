from __future__ import annotations
from typing import Any

from .tensor_spec import TensorSpec
from .builder import Builder
from .layers.base import Layer

class Model:
    def __init__(self, dtype: str = "f16", device: str = "cuda"):
        self.dtype = str(dtype)
        self.device = str(device)
        self.b = Builder(dtype=self.dtype, device=self.device)
    
    def input(self, name: str, spec: TensorSpec) -> int:
        if spec.dtype is None or spec.device is None:
            spec = TensorSpec(
                shape=spec.shape,
                dtype=spec.dtype or self.dtype,
                device=spec.device or self.device,
            )
        return self.b.input(name, spec)



    def add(self, layer: Layer, *xs: int) -> int | tuple[int, ...]:
        return layer.emit(self.b, *xs)

    def output(self, name: str, vid: int) -> None:
        self.b.output(name, vid)

    def dump(self) -> str:
        b = self.b

        def vfmt(vid: int) -> str:
            v = b.values[vid]
            s = v.spec
            shp = ",".join(str(x) for x in s.shape)
            return f"{v.name}:{s.dtype}[{shp}]"

        in_names = [b.values[v].name for v in b.input_vids]
        out_names = [b.values[v].name for v in b.output_vids]

        lines = []
        lines.append(f"[Graph] inputs={in_names} outputs={out_names} ops={len(b.ops)}")

        for i, op in enumerate(b.ops, start=1):
            ins = ", ".join(b.values[v].name for v in op.inputs)
            outs = ", ".join(b.values[v].name for v in op.outputs)

            # attrs는 너무 길어질 수 있어서 한 줄 유지
            extras = []
            if op.attrs:
                extras.append(f"attrs={op.attrs}")
            if op.constraints:
                extras.append(f"constraints={op.constraints}")
            if op.saved:
                saved_names = [b.values[v].name for v in op.saved]
                extras.append(f"saved={saved_names}")
            extra_str = ("  " + " ".join(extras)) if extras else ""

            # 핵심 출력: op + inputs/outputs
            lines.append(f"  o{i}: {op.kind}({ins}) -> {outs}{extra_str}")

        return "\n".join(lines)
