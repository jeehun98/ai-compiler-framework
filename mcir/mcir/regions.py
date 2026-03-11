from __future__ import annotations

from dataclasses import dataclass, field

from .nodes import MCNode
from .values import MCValue


@dataclass
class Region:
    name: str
    kind: str
    inputs: list[MCValue] = field(default_factory=list)
    outputs: list[MCValue] = field(default_factory=list)
    nodes: list[MCNode] = field(default_factory=list)
    subregions: list["Region"] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)


@dataclass
class ExecutionRegion(Region):
    def __init__(self, name: str):
        super().__init__(name=name, kind="execution")


@dataclass
class StreamingRegion(Region):
    def __init__(self, name: str, stream_axis: str = "sequence"):
        super().__init__(name=name, kind="streaming")
        self.attrs["stream_axis"] = stream_axis


@dataclass
class TileRegion(Region):
    def __init__(self, name: str, tile_m: int, tile_n: int, tile_k: int):
        super().__init__(name=name, kind="tile")
        self.attrs["tile_m"] = tile_m
        self.attrs["tile_n"] = tile_n
        self.attrs["tile_k"] = tile_k