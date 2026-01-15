# examples/python/aicf_fw/core_v2/cuda_exec.py
#
# Python-side IR executable for AICF CUDA bindings.
#
# - Input format: SAME "values/nodes" JSON schema you used in C++ bindings.
# - Supports:
#   1) already-lowered nodes: "gemm", "bias_add", "relu", "copy", "reduce_sum", "relu_bwd", ...
#   2) high-level IR nodes: "Linear", "ReLU", "MseGrad", "GradZero", "StepInc", "BiasCorr", "AdamStep"
# - Lazily allocates intermediates from meta ("values").
# - External tensors must be bound via exe.bind(name, tensor).
# - Execution uses ONLY _C.op_call(kind, inputs, outputs, attrs, stream).
# - CUDA Graph wrapper:
#     s = _C.graph_begin(); exe.run_once(stream=s); _C.graph_end(); _C.graph_launch()
#
# Usage:
#   exe = AICFCudaExecutable.from_json_str(ir_json)
#   for name in exe.required_inputs: exe.bind(name, tensor)
#   exe.run_once()
#   exe.capture()
#   exe.replay(n=10)

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# NOTE: your package layout is build/python/aicf_cuda
# so typical import is:
#   from aicf_cuda import _C
from aicf_cuda import _C


# -----------------------------
# Helpers: normalize strings
# -----------------------------

def _to_lower(s: str) -> str:
    return s.lower().strip()


def _parse_torch_dtype(dt: str) -> torch.dtype:
    s = _to_lower(dt)
    if "float32" in s or "f32" in s:
        return torch.float32
    if "float16" in s or "f16" in s or "half" in s:
        return torch.float16
    if "bfloat16" in s or "bf16" in s:
        return torch.bfloat16
    if "int32" in s or "i32" in s:
        return torch.int32
    if "int64" in s or "i64" in s:
        return torch.int64
    raise ValueError(f"unsupported dtype string: {dt}")


def _parse_torch_device(dev: str) -> torch.device:
    s = _to_lower(dev)
    if "cuda" in s:
        # accept "cuda", "cuda:0", "CUDA:1"
        if ":" in s:
            idx = int(s.split(":")[1])
        else:
            idx = 0
        return torch.device("cuda", idx)
    if "cpu" in s:
        return torch.device("cpu")
    raise ValueError(f"unsupported device string: {dev}")


def _ensure_attr_types(attrs: Dict[str, Any]) -> Dict[str, Any]:
    # C++ AttrPack supports bool/int/float only.
    out: Dict[str, Any] = {}
    for k, v in (attrs or {}).items():
        if isinstance(v, (bool, int, float)):
            out[k] = v
        else:
            raise TypeError(f"attrs[{k}] must be bool/int/float, got {type(v)}")
    return out


# -----------------------------
# OpKind mapping (string -> int)
# -----------------------------

# We use ints so we don't depend on enum objects at runtime.
# These names must match your C++ opkind_to_name() mapping.
_LOWERED_NAME_TO_KIND: Dict[str, int] = {
    "add": int(_C.OpKind.EltwiseAdd),
    "relu": int(_C.OpKind.EltwiseRelu),
    "gemm": int(_C.OpKind.Gemm),
    "bias_add": int(_C.OpKind.BiasAdd),
    "reduce_sum": int(_C.OpKind.ReduceSum),
    "mse_grad": int(_C.OpKind.MseGrad),
    "relu_bwd": int(_C.OpKind.ReluBwd),
    "sgd_step": int(_C.OpKind.SgdStep),
    "copy": int(_C.OpKind.Copy),
    "grad_zero": int(_C.OpKind.GradZero),
    "adam_step": int(_C.OpKind.AdamStep),
    "step_inc": int(_C.OpKind.StepInc),
    "bias_corr": int(_C.OpKind.BiasCorr),
    # layernorm/batchnorm kinds exist but not used by lowering rules below
    "layernorm_fwd": int(_C.OpKind.LayerNormFwd),
    "layernorm_bwd": int(_C.OpKind.LayerNormBwd),
    "batchnorm_fwd": int(_C.OpKind.BatchNormFwd),
    "batchnorm_bwd": int(_C.OpKind.BatchNormBwd),
}

# names accepted as "already-lowered"
_ACCEPT_AS_LOWERED = set(_LOWERED_NAME_TO_KIND.keys())


@dataclass(frozen=True)
class ValMeta:
    shape: List[int]
    dtype: str
    device: str


@dataclass
class LoweredOp:
    kind: int
    ins: List[str]
    outs: List[str]
    attrs: Dict[str, Any]


# -----------------------------
# Lowering rules (Python-side)
# -----------------------------

def _lowered_kind_from_name(op: str) -> int:
    s = _to_lower(op)
    if s not in _LOWERED_NAME_TO_KIND:
        raise ValueError(f"unknown lowered op name: {op}")
    return _LOWERED_NAME_TO_KIND[s]


def lower_ir_node_into(
    ir_op: str,
    in_names: List[str],
    out_names: List[str],
    attrs: Dict[str, Any],
) -> List[LoweredOp]:
    """
    Returns a list of LoweredOp.
    - Accepts already-lowered ops as-is.
    - Lowers selected high-level IR ops into lowered ops.
    """
    op = _to_lower(ir_op)
    attrs = _ensure_attr_types(attrs)

    # 1) already-lowered: pass-through
    if op in _ACCEPT_AS_LOWERED:
        return [LoweredOp(
            kind=_lowered_kind_from_name(op),
            ins=list(in_names),
            outs=list(out_names),
            attrs=dict(attrs),
        )]

    # 2) high-level lowering rules
    if op == "linear":
        # inputs [x, W, (b)] outputs [y]
        # gemm(x, W) -> y  with transB=True
        # optional bias_add(y, b) -> y in-place
        if len(out_names) != 1:
            raise ValueError("Linear: expects 1 output")
        if len(in_names) not in (2, 3):
            raise ValueError("Linear: expects 2 or 3 inputs [x, W, (b)]")

        y = out_names[0]
        ops: List[LoweredOp] = []
        ops.append(LoweredOp(
            kind=_lowered_kind_from_name("gemm"),
            ins=[in_names[0], in_names[1]],
            outs=[y],
            attrs={"transB": True},
        ))

        has_bias = False
        if "bias" in attrs:
            has_bias = bool(attrs["bias"])
        elif len(in_names) == 3:
            has_bias = True

        if has_bias:
            if len(in_names) < 3:
                raise ValueError("Linear: bias=true but no bias input")
            ops.append(LoweredOp(
                kind=_lowered_kind_from_name("bias_add"),
                ins=[y, in_names[2]],
                outs=[y],  # in-place
                attrs={},
            ))
        return ops

    if op == "relu":
        if len(in_names) != 1 or len(out_names) != 1:
            raise ValueError("ReLU: expects 1 in/1 out")
        return [LoweredOp(
            kind=_lowered_kind_from_name("relu"),
            ins=[in_names[0]],
            outs=[out_names[0]],
            attrs={},
        )]

    if op in ("msegrad", "mse_grad"):
        if len(in_names) != 2 or len(out_names) != 1:
            raise ValueError("MseGrad: expects 2 in/1 out")
        a: Dict[str, Any] = {}
        if "scale" in attrs:
            a["scale"] = attrs["scale"]
        return [LoweredOp(
            kind=_lowered_kind_from_name("mse_grad"),
            ins=[in_names[0], in_names[1]],
            outs=[out_names[0]],
            attrs=a,
        )]

    if op in ("gradzero", "grad_zero"):
        if len(in_names) != 1 or len(out_names) != 1:
            raise ValueError("GradZero: expects 1 in/1 out")
        return [LoweredOp(
            kind=_lowered_kind_from_name("grad_zero"),
            ins=[in_names[0]],
            outs=[out_names[0]],
            attrs={},
        )]

    if op in ("stepinc", "step_inc"):
        if len(in_names) != 1 or len(out_names) != 1:
            raise ValueError("StepInc: expects 1 in/1 out")
        return [LoweredOp(
            kind=_lowered_kind_from_name("step_inc"),
            ins=[in_names[0]],
            outs=[out_names[0]],
            attrs={},
        )]

    if op in ("biascorr", "bias_corr"):
        if len(in_names) != 1 or len(out_names) != 2:
            raise ValueError("BiasCorr: expects 1 in/2 out")
        a: Dict[str, Any] = {}
        if "beta1" in attrs:
            a["beta1"] = attrs["beta1"]
        if "beta2" in attrs:
            a["beta2"] = attrs["beta2"]
        return [LoweredOp(
            kind=_lowered_kind_from_name("bias_corr"),
            ins=[in_names[0]],
            outs=[out_names[0], out_names[1]],
            attrs=a,
        )]

    if op in ("adamstep", "adam_step"):
        if len(in_names) != 6 or len(out_names) != 3:
            raise ValueError("AdamStep: expects 6 in/3 out")
        a: Dict[str, Any] = {}
        for k in ("lr", "beta1", "beta2", "eps"):
            if k in attrs:
                a[k] = attrs[k]
        return [LoweredOp(
            kind=_lowered_kind_from_name("adam_step"),
            ins=list(in_names),
            outs=list(out_names),
            attrs=a,
        )]

    if op == "backward":
        raise ValueError("Backward is not supported in this Python exec (yet).")

    raise ValueError(f"unsupported IR op: {ir_op}")


# -----------------------------
# Executable
# -----------------------------

class AICFCudaExecutable:
    def __init__(
        self,
        graph_name: str,
        ops: List[LoweredOp],
        meta: Dict[str, ValMeta],
    ):
        self.graph_name = graph_name
        self.ops = ops

        # name -> meta (for lazy alloc)
        self.meta: Dict[str, ValMeta] = dict(meta)

        # name -> bound tensor (externals + allocated)
        self.bind: Dict[str, torch.Tensor] = {}

        # computed at compile time
        self.required_inputs: List[str] = self._compute_required_inputs()

        # capture state (singleton graph in C++; keep a bool here for UX only)
        self._captured: bool = False

    # -------- compile from JSON --------

    @staticmethod
    def from_json_str(ir_json: str) -> "AICFCudaExecutable":
        root = json.loads(ir_json)
        return AICFCudaExecutable.from_dict(root)

    @staticmethod
    def from_dict(root: Dict[str, Any]) -> "AICFCudaExecutable":
        graph_name = root.get("graph", "graph")

        if "values" not in root:
            raise ValueError("missing 'values'")
        values: Dict[str, Any] = root["values"]

        # Build id -> name mapping and meta
        max_id = -1
        for _, v in values.items():
            vid = int(v["id"])
            if vid > max_id:
                max_id = vid

        id2name: List[Optional[str]] = [None] * (max_id + 1)
        id_valid: List[bool] = [False] * (max_id + 1)

        meta: Dict[str, ValMeta] = {}
        for _, v in values.items():
            vid = int(v["id"])
            name = str(v["name"])
            id2name[vid] = name
            id_valid[vid] = True

            shape = [int(x) for x in v["shape"]]
            dtype = str(v["dtype"])
            device = str(v["device"])
            meta[name] = ValMeta(shape=shape, dtype=dtype, device=device)

        if "nodes" not in root:
            raise ValueError("missing 'nodes'")
        nodes: List[Dict[str, Any]] = root["nodes"]

        ops: List[LoweredOp] = []

        for n in nodes:
            op_name = str(n["op"])
            ins = list(n.get("inputs", []))
            outs = list(n.get("outputs", []))
            attrs = dict(n.get("attrs", {}))

            in_names: List[str] = []
            out_names: List[str] = []

            for vid in ins:
                vid = int(vid)
                if vid < 0 or vid >= len(id2name) or not id_valid[vid]:
                    raise ValueError(f"invalid input vid={vid}")
                in_names.append(id2name[vid] or "")

            for vid in outs:
                vid = int(vid)
                if vid < 0 or vid >= len(id2name) or not id_valid[vid]:
                    raise ValueError(f"invalid output vid={vid}")
                out_names.append(id2name[vid] or "")

            lowered = lower_ir_node_into(op_name, in_names, out_names, attrs)
            ops.extend(lowered)

        return AICFCudaExecutable(graph_name=graph_name, ops=ops, meta=meta)

    # -------- binding / alloc --------

    def bind_tensor(self, name: str, t: torch.Tensor) -> None:
        if not isinstance(t, torch.Tensor):
            raise TypeError("bind_tensor expects a torch.Tensor")
        self.bind[name] = t

    def get_or_alloc(self, name: str) -> torch.Tensor:
        if name in self.bind:
            return self.bind[name]
        if name not in self.meta:
            raise KeyError(f"tensor not bound and no meta: name={name}")

        m = self.meta[name]
        dt = _parse_torch_dtype(m.dtype)
        dv = _parse_torch_device(m.device)

        t = torch.empty(list(m.shape), device=dv, dtype=dt)
        self.bind[name] = t
        return t

    def _compute_required_inputs(self) -> List[str]:
        produced = set()
        used = set()
        for op in self.ops:
            for nm in op.ins:
                used.add(nm)
            for nm in op.outs:
                produced.add(nm)

        req = sorted([nm for nm in used if nm not in produced])
        return req

    # -------- execute --------

    def run_once(self, stream: int = 0) -> None:
        """
        Execute ops eagerly once.
        - If stream != 0, passes that cudaStream_t handle into C++ op_call.
        """
        for op in self.ops:
            in_tensors = [self.get_or_alloc(nm) for nm in op.ins]
            out_tensors = [self.get_or_alloc(nm) for nm in op.outs]
            _C.op_call(int(op.kind), in_tensors, out_tensors, dict(op.attrs), stream=int(stream))

    # -------- graph capture / replay --------

    def capture(self) -> None:
        """
        Capture current executable ops into CUDA Graph (C++ singleton graph).
        IMPORTANT: capture uses dedicated stream returned by graph_begin().
        """
        s = int(_C.graph_begin())
        self.run_once(stream=s)
        _C.graph_end()
        self._captured = True

    def replay(self, n: int = 1) -> None:
        if n <= 0:
            return
        if not self._captured:
            # allow replay even if bool not set, but raise to catch misuse
            raise RuntimeError("replay called before capture()")
        for _ in range(int(n)):
            _C.graph_launch()

    def reset_graph(self) -> None:
        _C.graph_reset()
        self._captured = False

    # -------- debug helpers --------

    def dump(self) -> Dict[str, Any]:
        return {
            "graph": self.graph_name,
            "ops": [
                {
                    "kind": int(op.kind),
                    "inputs": list(op.ins),
                    "outputs": list(op.outs),
                    "attrs": dict(op.attrs),
                }
                for op in self.ops
            ],
            "bound": sorted(list(self.bind.keys())),
            "required_inputs": list(self.required_inputs),
        }
