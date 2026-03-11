from __future__ import annotations

from .module import MCModule
from .nodes import MCNode
from .regions import ExecutionRegion, StreamingRegion, TileRegion
from .values import MCValue


class MCIRBuilder:
    def module(self) -> MCModule:
        return MCModule()

    def value(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: str,
        residency: str = "global",
    ) -> MCValue:
        return MCValue(
            name=name,
            shape=shape,
            dtype=dtype,
            residency=residency,
        )

    def node(
        self,
        name: str,
        op: str,
        inputs: list[MCValue] | None = None,
        outputs: list[MCValue] | None = None,
        **attrs,
    ) -> MCNode:
        return MCNode(
            name=name,
            op=op,
            inputs=inputs or [],
            outputs=outputs or [],
            attrs=attrs,
        )

    def execution_region(self, name: str) -> ExecutionRegion:
        return ExecutionRegion(name)

    def streaming_region(self, name: str, stream_axis: str = "sequence") -> StreamingRegion:
        return StreamingRegion(name, stream_axis=stream_axis)

    def tile_region(self, name: str, tile_m: int, tile_n: int, tile_k: int) -> TileRegion:
        return TileRegion(name, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)