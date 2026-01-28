from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Protocol, Union


# -------------------------------------------------
# Tensor descriptor (lightweight, additive only)
# -------------------------------------------------
@dataclass(frozen=True)
class TensorDesc:
    shape: Tuple[int, ...]
    dtype: str
    strides: Optional[Tuple[int, ...]] = None
    is_contiguous: Optional[bool] = None

    @staticmethod
    def from_any(x: Any) -> "TensorDesc":
        """
        Accepts:
          - torch.Tensor
          - dict-like: {"shape": (..), "dtype": "...", "strides": (..), "is_contiguous": bool}
          - object with attributes .shape/.dtype/.stride()/.is_contiguous()
        """
        # torch.Tensor
        if hasattr(x, "shape") and hasattr(x, "dtype") and callable(getattr(x, "stride", None)):
            shape = tuple(int(v) for v in x.shape)
            dtype = str(x.dtype).replace("torch.", "")
            strides = tuple(int(v) for v in x.stride())
            is_contig = bool(x.is_contiguous()) if hasattr(x, "is_contiguous") else None
            return TensorDesc(
                shape=shape,
                dtype=dtype,
                strides=strides,
                is_contiguous=is_contig,
            )

        # dict-like
        if isinstance(x, dict):
            shape = tuple(int(v) for v in x.get("shape", ()))
            dtype = str(x.get("dtype", "unknown"))
            strides = x.get("strides", None)
            if strides is not None:
                strides = tuple(int(v) for v in strides)
            is_contig = x.get("is_contiguous", None)
            if is_contig is not None:
                is_contig = bool(is_contig)
            return TensorDesc(
                shape=shape,
                dtype=dtype,
                strides=strides,
                is_contiguous=is_contig,
            )

        # generic object with fields
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            shape = tuple(int(v) for v in getattr(x, "shape"))
            dtype = str(getattr(x, "dtype"))
            strides = getattr(x, "strides", None)
            if strides is not None:
                strides = tuple(int(v) for v in strides)
            is_contig = getattr(x, "is_contiguous", None)
            if is_contig is not None:
                is_contig = bool(is_contig)
            return TensorDesc(
                shape=shape,
                dtype=dtype,
                strides=strides,
                is_contiguous=is_contig,
            )

        # fallback
        return TensorDesc(shape=tuple(), dtype="unknown")


# -------------------------------------------------
# Op Attribute (purely additive metadata)
# -------------------------------------------------
@dataclass
class OpAttr:
    op_kind: str
    op_id: Optional[int] = None

    # value ids
    inputs: List[int] = field(default_factory=list)
    outputs: List[int] = field(default_factory=list)

    # semantic-ish fields (no compose yet)
    sig: Optional[str] = None
    dtypes: Dict[str, str] = field(default_factory=dict)          # in0/out0/...
    shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    layout: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    # trace-only info
    kid: Optional[str] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------------------------------
# LoweredOp normalization view
# -------------------------------------------------
@dataclass(frozen=True)
class LoweredOpView:
    kind: str
    inputs: List[int]
    outputs: List[int]
    attrs: Dict[str, Any]
    op_id: Optional[int] = None
    kid: Optional[str] = None

    @staticmethod
    def from_any(op: Any, op_id: Optional[int] = None) -> "LoweredOpView":
        # dict-style
        if isinstance(op, dict):
            return LoweredOpView(
                kind=str(op.get("kind") or op.get("op_kind") or op.get("name")),
                inputs=list(op.get("inputs", op.get("ins", []))),
                outputs=list(op.get("outputs", op.get("outs", []))),
                attrs=dict(op.get("attrs", op.get("params", {})) or {}),
                op_id=op.get("id", op_id),
                kid=op.get("kid", None),
            )

        # object-style
        kind = (
            getattr(op, "kind", None)
            or getattr(op, "op_kind", None)
            or getattr(op, "name", None)
        )
        if kind is None:
            kind = op.__class__.__name__

        inputs = list(getattr(op, "inputs", getattr(op, "ins", [])))
        outputs = list(getattr(op, "outputs", getattr(op, "outs", [])))
        attrs = dict(getattr(op, "attrs", getattr(op, "params", {})) or {})
        kid = getattr(op, "kid", None)
        oid = getattr(op, "id", op_id)

        return LoweredOpView(
            kind=str(kind),
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            op_id=oid,
            kid=kid,
        )


# -------------------------------------------------
# Common helpers
# -------------------------------------------------
ValueDescs = Union[Dict[int, Any], List[Any]]


def get_desc(value_descs: ValueDescs, vid: int) -> TensorDesc:
    if isinstance(value_descs, dict):
        return TensorDesc.from_any(value_descs.get(vid, {}))
    if 0 <= vid < len(value_descs):
        return TensorDesc.from_any(value_descs[vid])
    return TensorDesc(shape=tuple(), dtype="unknown")


def fill_common(attr: OpAttr, v: LoweredOpView, value_descs: ValueDescs) -> None:
    # inputs
    for i, vid in enumerate(v.inputs):
        td = get_desc(value_descs, vid)
        attr.shapes[f"in{i}"] = td.shape
        attr.dtypes[f"in{i}"] = td.dtype
        if td.is_contiguous is not None:
            attr.layout[f"contig_in{i}"] = td.is_contiguous
        if td.strides is not None:
            attr.layout[f"strides_in{i}"] = td.strides

    # outputs
    for i, vid in enumerate(v.outputs):
        td = get_desc(value_descs, vid)
        attr.shapes[f"out{i}"] = td.shape
        attr.dtypes[f"out{i}"] = td.dtype
        if td.is_contiguous is not None:
            attr.layout[f"contig_out{i}"] = td.is_contiguous
        if td.strides is not None:
            attr.layout[f"strides_out{i}"] = td.strides


# -------------------------------------------------
# Builder protocol (op-specific files implement this)
# -------------------------------------------------
class OpAttrBuilder(Protocol):
    kind: str

    def build(
        self,
        op: Any,
        value_descs: ValueDescs,
        op_id: Optional[int] = None,
    ) -> OpAttr:
        ...


