from __future__ import annotations

from .module import MCModule


_VALID_RESIDENCY = {"global", "shared", "register"}
_VALID_REGION_KINDS = {"execution", "streaming", "tile"}


def validate_module(module: MCModule) -> None:
    for region in module.regions:
        _validate_region(region)


def _validate_region(region) -> None:
    if not region.name:
        raise ValueError("Region name must not be empty")
    if region.kind not in _VALID_REGION_KINDS:
        raise ValueError(f"Invalid region kind: {region.kind}")

    for value in region.inputs + region.outputs:
        _validate_value(value)

    for node in region.nodes:
        if not node.name:
            raise ValueError("Node name must not be empty")
        if not node.op:
            raise ValueError("Node op must not be empty")
        for value in node.inputs + node.outputs:
            _validate_value(value)

    for sub in region.subregions:
        _validate_region(sub)


def _validate_value(value) -> None:
    if not value.name:
        raise ValueError("Value name must not be empty")
    if not value.shape:
        raise ValueError(f"Value {value.name} has empty shape")
    if not value.dtype:
        raise ValueError(f"Value {value.name} has empty dtype")
    if value.residency not in _VALID_RESIDENCY:
        raise ValueError(f"Invalid residency {value.residency} for value {value.name}")