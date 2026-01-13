# aicf_fw/core/compile.py
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set, Iterable, Callable

import torch

from .ir import IRGraph, IRValue
from .runtime import warmup_capture_safe
from aicf_fw.backend import get_backend


# ============================================================
# CompileArtifact
# ============================================================

@dataclass
class CompileArtifact:
    """
    Compilation artifact produced by compile/lower/capture.

    - ir: compiled IR object
    - lowered: list of backend ops (dicts)
    - trace_ops: runtime trace op names captured during CUDA graph capture
    - backend: backend instance (must support replay())
    - env: IRValue vid -> torch.Tensor runtime binding (for IRExecutor)
    - env_provider: optional callable returning fresh env (for restore/alias changes)
    """
    name: str
    ir: Any
    lowered: List[Dict[str, Any]]
    trace_ops: List[str]
    backend: Any
    env: Dict[int, torch.Tensor] = field(default_factory=dict)

    # NEW: provider for lazy/refreshable env
    env_provider: Optional[Callable[[], Dict[int, torch.Tensor]]] = None

    def attach_env(self, env: Dict[int, torch.Tensor], *, merge: bool = False) -> None:
        if env is None:
            raise RuntimeError("CompileArtifact.attach_env: env is None")

        norm: Dict[int, torch.Tensor] = {}
        for k, v in env.items():
            vid = int(k)
            if not isinstance(v, torch.Tensor):
                raise RuntimeError(f"CompileArtifact.attach_env: env[{vid}] is not torch.Tensor: {type(v)}")
            norm[vid] = v

        if merge:
            self.env.update(norm)
        else:
            self.env = norm

    # NEW
    def attach_env_provider(self, provider: Callable[[], Dict[int, torch.Tensor]]) -> None:
        if provider is None or not callable(provider):
            raise RuntimeError("CompileArtifact.attach_env_provider: provider must be callable")
        self.env_provider = provider

    def runtime_env(self) -> Dict[int, torch.Tensor]:
        """
        Provider 우선:
          - provider가 있으면 매 호출마다 fresh env 반환
          - 없으면 self.env 반환
        """
        if self.env_provider is not None:
            env = self.env_provider()
            if not isinstance(env, dict) or len(env) == 0:
                raise RuntimeError("CompileArtifact.runtime_env: env_provider returned empty env")
            return env
        return self.env

    def has_env(self) -> bool:
        if self.env_provider is not None:
            return True
        return isinstance(self.env, dict) and len(self.env) > 0

    def assert_trace_has(self, op: str) -> None:
        if op not in self.trace_ops:
            raise AssertionError(f"[trace] {op} missing in captured runtime ops: {self.trace_ops}")

    def assert_runtime_matches_lowering(self, model: Any = None, *, trace_filter: bool = True) -> None:
        """
        Ensure runtime-captured ops contain all lowered ops (optionally filtered).

        NOTE:
          - Runtime trace comes from C++ OpKind names. For Copy it always records as "copy".
          - Therefore semantic copy variants must NOT be validated by op-name.
          - We treat "copy" as auxiliary in trace validation to avoid environment-dependent noise.
        """
        if not isinstance(self.trace_ops, list) or len(self.trace_ops) == 0:
            raise AssertionError("[trace] empty trace_ops. Did you run compile_and_capture(trace=True)?")

        lowered_ops = [it["op"] for it in self.lowered]

        if trace_filter:
            IGNORE = {"copy"}
            lowered_ops = [op for op in lowered_ops if op not in IGNORE and not op.startswith("UNSUPPORTED<")]

        missing = [op for op in lowered_ops if op not in self.trace_ops]
        if missing:
            tail = self.trace_ops[-64:] if len(self.trace_ops) > 64 else self.trace_ops
            raise AssertionError(
                "[lowering/trace] runtime trace missing lowered ops:\n"
                + "\n".join([f"  - {op}" for op in missing[:64]])
                + "\n"
                + f"trace_tail({len(tail)}): {tail}"
            )


# ============================================================
# Semantic lowering validation (IR/lowered structure)
# ============================================================

def _assert_lowering_has_semantic_saved_copy(ir: IRGraph, lowered: List[Dict[str, Any]]) -> None:
    saved_vids = [
        int(vid) for vid, val in ir.values.items()
        if str(getattr(val, "name", "")) == "relu_y_saved"
    ]
    if not saved_vids:
        raise AssertionError("[semantic] missing IRValue named 'relu_y_saved'")
    saved_set = set(saved_vids)

    copy_into_saved = False
    for it in lowered:
        if it.get("op") != "copy":
            continue
        outs = list(it.get("outputs", []))
        if outs and int(outs[0]) in saved_set:
            copy_into_saved = True
            break
    if not copy_into_saved:
        raise AssertionError("[semantic] missing lowering: copy -> relu_y_saved")

    relu_bwd_uses_saved = False
    for it in lowered:
        if it.get("op") != "relu_bwd":
            continue
        ins = list(it.get("inputs", []))
        if len(ins) >= 2 and int(ins[1]) in saved_set:
            relu_bwd_uses_saved = True
            break
    if not relu_bwd_uses_saved:
        raise AssertionError("[semantic] relu_bwd does not consume relu_y_saved as 2nd input")


# ============================================================
# Tracing
# ============================================================

_TRACING = False
_IR: Optional[IRGraph] = None

_TRACE_VAL_CACHE_OBJ: Dict[int, IRValue] = {}
_TRACE_VAL_CACHE_TORCH: Dict[int, IRValue] = {}


def is_tracing() -> bool:
    return bool(_TRACING)


def get_ir() -> IRGraph:
    if _IR is None:
        raise RuntimeError("IRBuilder is not set. Use `with tracing(ir): ...`")
    return _IR


def trace_reset_cache() -> None:
    _TRACE_VAL_CACHE_OBJ.clear()
    _TRACE_VAL_CACHE_TORCH.clear()


def _torch_key(x: torch.Tensor) -> int:
    try:
        return int(x.data_ptr())
    except Exception:
        return id(x)


def as_ir_value_obj(obj: Any, *, name: str, shape, dtype, device) -> IRValue:
    if not is_tracing():
        raise RuntimeError("as_ir_value_obj() called outside tracing")

    k = id(obj)
    v = _TRACE_VAL_CACHE_OBJ.get(k)
    if v is not None:
        return v

    ir = get_ir()
    v = ir.new_value(name=name, shape=tuple(shape), dtype=str(dtype), device=str(device))
    _TRACE_VAL_CACHE_OBJ[k] = v
    return v


def as_ir_value_torch(x: torch.Tensor, *, name: str) -> IRValue:
    if not is_tracing():
        raise RuntimeError("as_ir_value_torch() called outside tracing")

    k = _torch_key(x)
    v = _TRACE_VAL_CACHE_TORCH.get(k)
    if v is not None:
        return v

    ir = get_ir()
    v = ir.new_value(name=name, shape=tuple(x.shape), dtype=str(x.dtype), device=str(x.device))
    _TRACE_VAL_CACHE_TORCH[k] = v
    return v


@contextmanager
def tracing(ir: IRGraph):
    global _TRACING, _IR
    prev_t = _TRACING
    prev_ir = _IR

    _TRACING = True
    _IR = ir
    trace_reset_cache()

    try:
        yield ir
    finally:
        _TRACING = prev_t
        _IR = prev_ir
        trace_reset_cache()


# ============================================================
# IR compile
# ============================================================

def compile_ir(step_fn, *, name: str = "train_step") -> IRGraph:
    ir = IRGraph(name=name)
    with tracing(ir):
        step_fn()
    return ir


# ============================================================
# Lowering
# (너가 올린 lower_to_backend_ops 그대로 두면 됨)
# ============================================================

# !!! 여기서는 지면상 생략하지 않고 너가 쓰던 lower_to_backend_ops / autobind_env_from_lowered를 그대로 유지하면 됨.
# 이미 올려준 버전을 그대로 붙여넣어 사용해.
from typing import cast
# (주의) 실제 파일에서는 위 주석 대신, 네가 쓰는 lower_to_backend_ops, autobind_env_from_lowered 전체가 있어야 함.


# ============================================================
# compile + capture (single capture)
# ============================================================

def compile_and_capture(
    step_fn,
    *,
    name: str = "train_step",
    warmup_runs: int = 2,
    warmup_sync: bool = True,
    validate: bool = True,
    trace: bool = True,
    enforce_ops: Sequence[str] = ("adam_step",),
    torch_sync: bool = True,
    bind_model: Optional[Any] = None,
    bind_optim: Optional[Any] = None,
    bind_x: Optional[Any] = None,
    bind_t: Optional[Any] = None,
    autobind_env: bool = True,
) -> CompileArtifact:
    ir = compile_ir(step_fn, name=name)
    lowered = lower_to_backend_ops(ir)

    if validate:
        _assert_lowering_has_semantic_saved_copy(ir, lowered)

    # ---- warmup: NO state drift ----
    prev_warm = os.getenv("AICF_WARMUP", None)
    if warmup_runs and warmup_runs > 0:
        os.environ["AICF_WARMUP"] = "1"
        try:
            warmup_capture_safe(train_step=step_fn, runs=warmup_runs, sync=warmup_sync)
        finally:
            if prev_warm is None:
                os.environ.pop("AICF_WARMUP", None)
            else:
                os.environ["AICF_WARMUP"] = prev_warm

    backend = get_backend()

    backend.capture_reset()
    if torch_sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    backend.trace_reset()
    backend.trace_enable(bool(trace))

    # ---- capture exactly once; updates enabled ----
    prev_cap = os.getenv("AICF_WARMUP", None)
    os.environ["AICF_WARMUP"] = "0"
    try:
        backend.capture_begin()
        step_fn()
        backend.capture_end()
    finally:
        if prev_cap is None:
            os.environ.pop("AICF_WARMUP", None)
        else:
            os.environ["AICF_WARMUP"] = prev_cap

    if torch_sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    trace_ops: List[str] = backend.trace_get() if trace else []
    backend.trace_enable(False)

    art = CompileArtifact(name=name, ir=ir, lowered=lowered, trace_ops=trace_ops, backend=backend)

    for op in enforce_ops:
        art.assert_trace_has(op)

    if autobind_env:
        env = autobind_env_from_lowered(ir, lowered, env=art.env)
        art.attach_env(env, merge=False)

    return art


__all__ = [
    "CompileArtifact",
    "compile_ir",
    "lower_to_backend_ops",
    "compile_and_capture",
    "autobind_env_from_lowered",
    "tracing",
    "is_tracing",
    "get_ir",
    "as_ir_value_obj",
    "as_ir_value_torch",
    "trace_reset_cache",
]


# ============================================================
# Semantic lowering validation (IR/lowered structure)
# ============================================================

def _assert_lowering_has_semantic_saved_copy(ir: IRGraph, lowered: List[Dict[str, Any]]) -> None:
    """
    Semantic validation for ReLU backward dependency:
      - There must exist a copy op that writes into a value named 'relu_y_saved'
      - There must exist a relu_bwd op that consumes that same vid as its 2nd input

    This avoids relying on runtime trace op-name (copy_saved vs copy).
    """
    saved_vids = [
        int(vid) for vid, val in ir.values.items()
        if str(getattr(val, "name", "")) == "relu_y_saved"
    ]
    if not saved_vids:
        raise AssertionError("[semantic] missing IRValue named 'relu_y_saved'")
    saved_set = set(saved_vids)

    # copy -> relu_y_saved
    copy_into_saved = False
    for it in lowered:
        if it.get("op") != "copy":
            continue
        outs = list(it.get("outputs", []))
        if outs and int(outs[0]) in saved_set:
            copy_into_saved = True
            break
    if not copy_into_saved:
        raise AssertionError("[semantic] missing lowering: copy -> relu_y_saved")

    # relu_bwd uses relu_y_saved as 2nd input
    relu_bwd_uses_saved = False
    for it in lowered:
        if it.get("op") != "relu_bwd":
            continue
        ins = list(it.get("inputs", []))
        if len(ins) >= 2 and int(ins[1]) in saved_set:
            relu_bwd_uses_saved = True
            break
    if not relu_bwd_uses_saved:
        raise AssertionError("[semantic] relu_bwd does not consume relu_y_saved as 2nd input")


# ============================================================
# Tracing
# ============================================================

_TRACING = False
_IR: Optional[IRGraph] = None

_TRACE_VAL_CACHE_OBJ: Dict[int, IRValue] = {}
_TRACE_VAL_CACHE_TORCH: Dict[int, IRValue] = {}


def is_tracing() -> bool:
    return bool(_TRACING)


def get_ir() -> IRGraph:
    if _IR is None:
        raise RuntimeError("IRBuilder is not set. Use `with tracing(ir): ...`")
    return _IR


def trace_reset_cache() -> None:
    _TRACE_VAL_CACHE_OBJ.clear()
    _TRACE_VAL_CACHE_TORCH.clear()


def _torch_key(x: torch.Tensor) -> int:
    try:
        return int(x.data_ptr())
    except Exception:
        return id(x)


def as_ir_value_obj(obj: Any, *, name: str, shape, dtype, device) -> IRValue:
    if not is_tracing():
        raise RuntimeError("as_ir_value_obj() called outside tracing")

    k = id(obj)
    v = _TRACE_VAL_CACHE_OBJ.get(k)
    if v is not None:
        return v

    ir = get_ir()
    v = ir.new_value(name=name, shape=tuple(shape), dtype=str(dtype), device=str(device))
    _TRACE_VAL_CACHE_OBJ[k] = v
    return v


def as_ir_value_torch(x: torch.Tensor, *, name: str) -> IRValue:
    if not is_tracing():
        raise RuntimeError("as_ir_value_torch() called outside tracing")

    k = _torch_key(x)
    v = _TRACE_VAL_CACHE_TORCH.get(k)
    if v is not None:
        return v

    ir = get_ir()
    v = ir.new_value(name=name, shape=tuple(x.shape), dtype=str(x.dtype), device=str(x.device))
    _TRACE_VAL_CACHE_TORCH[k] = v
    return v


@contextmanager
def tracing(ir: IRGraph):
    global _TRACING, _IR
    prev_t = _TRACING
    prev_ir = _IR

    _TRACING = True
    _IR = ir
    trace_reset_cache()

    try:
        yield ir
    finally:
        _TRACING = prev_t
        _IR = prev_ir
        trace_reset_cache()


# ============================================================
# IR compile
# ============================================================

def compile_ir(step_fn, *, name: str = "train_step") -> IRGraph:
    ir = IRGraph(name=name)
    with tracing(ir):
        step_fn()
    return ir


# ============================================================
# Lowering
# ============================================================

def lower_to_backend_ops(ir) -> List[Dict[str, Any]]:
    """
    Lower IRGraph -> backend op list.

    Order:
      forward -> mse_grad -> backward ops -> (step_inc -> bias_corr -> adam_step * N)

    Critical fixes:
      - Backward GEMMs write into dedicated grad IRValues (name startswith "grad")
      - Never write GEMM outputs into leaf parameter.grad vids (prevents IRExec-only clobber)
      - AdamStep grad input must come from backward-produced grads.
      - Semantic copy variants are lowered as op="copy" (trace name is only "copy").
    """
    lowered: List[Dict[str, Any]] = []

    def v(vid: int):
        return ir.values[int(vid)]

    linear_nodes = [n for n in ir.nodes if n.op == "Linear"]
    relu_nodes   = [n for n in ir.nodes if n.op == "ReLU"]
    mse_nodes    = [n for n in ir.nodes if n.op == "MseGrad"]
    step_nodes   = [n for n in ir.nodes if n.op == "StepInc"]
    bc_nodes     = [n for n in ir.nodes if n.op == "BiasCorr"]
    adam_nodes   = [n for n in ir.nodes if n.op == "AdamStep"]
    bwd_nodes    = [n for n in ir.nodes if n.op == "Backward"]

    # --- grad pool allocator (name startswith "grad") ---
    used_grad: Set[int] = set()

    def _alloc_grad_vid(shape: Tuple[int, ...], *, like_vid: int, tag: str = "") -> int:
        ref = v(like_vid)
        nm = "grad" if tag == "" else f"grad{tag}"
        g = ir.new_value(
            name=nm,
            shape=tuple(shape),
            dtype=str(ref.dtype),
            device=str(ref.device),
        )
        return int(g.id)

    def pick_next_grad(shape: Tuple[int, ...], *, like_vid: int, tag: str = "") -> int:
        """
        Reuse only IRValues whose name starts with "grad".
        Prefers matching tag in name when possible.
        """
        shape = tuple(shape)
        want_prefix = "grad" if tag == "" else f"grad{tag}"

        # 1) exact tag+shape match
        cands1 = [
            int(vid) for vid, val in ir.values.items()
            if str(getattr(val, "name", "")).startswith(want_prefix) and tuple(val.shape) == shape
        ]
        for gid in cands1:
            if gid not in used_grad:
                used_grad.add(gid)
                return gid

        # 2) any grad* with same shape
        cands2 = [
            int(vid) for vid, val in ir.values.items()
            if str(getattr(val, "name", "")).startswith("grad") and tuple(val.shape) == shape
        ]
        for gid in cands2:
            if gid not in used_grad:
                used_grad.add(gid)
                return gid

        gid = _alloc_grad_vid(shape, like_vid=like_vid, tag=tag)
        used_grad.add(gid)
        return gid

    # -------------------------
    # 1) Forward lowering
    # -------------------------
    for n in ir.nodes:
        op = n.op

        if op == "Linear":
            x_vid = int(n.inputs[0])
            W_vid = int(n.inputs[1])
            y_vid = int(n.outputs[0])
            lowered.append({"op": "gemm", "attrs": {"transB": True}, "inputs": [x_vid, W_vid], "outputs": [y_vid]})
            if bool(n.attrs.get("bias", False)):
                b_vid = int(n.inputs[2])
                lowered.append({"op": "bias_add", "attrs": {}, "inputs": [y_vid, b_vid], "outputs": [y_vid]})
            continue

        if op == "ReLU":
            x_vid = int(n.inputs[0])
            y_vid = int(n.outputs[0])
            lowered.append({"op": "relu", "attrs": {}, "inputs": [x_vid], "outputs": [y_vid]})
            continue

        if op == "MseGrad":
            pred_vid = int(n.inputs[0])
            tgt_vid  = int(n.inputs[1])
            out_vid  = int(n.outputs[0])
            attrs: Dict[str, Any] = {}
            if "scale" in n.attrs:
                attrs["scale"] = float(n.attrs["scale"])
            lowered.append({"op": "mse_grad", "attrs": attrs, "inputs": [pred_vid, tgt_vid], "outputs": [out_vid]})
            continue

        if op in ("Backward", "StepInc", "BiasCorr", "AdamStep"):
            continue

        lowered.append({
            "op": f"UNSUPPORTED<{op}>",
            "attrs": dict(n.attrs),
            "inputs": list(n.inputs),
            "outputs": list(n.outputs),
        })

    # -------------------------
    # 2) Backward lowering
    # -------------------------
    grad_map_param_to_grad: Dict[int, int] = {}

    if bwd_nodes:
        if not mse_nodes or len(linear_nodes) < 2 or len(relu_nodes) < 1:
            raise RuntimeError("lower_to_backend_ops: Backward expects Linear->ReLU->Linear + MseGrad")

        lin0 = linear_nodes[0]
        relu0 = relu_nodes[0]
        lin1 = linear_nodes[1]
        mse = mse_nodes[-1]

        out_grad_vid = int(mse.outputs[0])  # dLoss/dPred

        x_vid = int(lin0.inputs[0])
        W0_vid = int(lin0.inputs[1])
        b0_vid = int(lin0.inputs[2]) if bool(lin0.attrs.get("bias", False)) else None
        lin0_out_vid = int(lin0.outputs[0])

        relu_out_vid = int(relu0.outputs[0])

        W1_vid = int(lin1.inputs[1])
        b1_vid = int(lin1.inputs[2]) if bool(lin1.attrs.get("bias", False)) else None

        # grads for params (dedicated grad vids)
        dW0_vid = pick_next_grad(tuple(v(W0_vid).shape), like_vid=W0_vid, tag="_W0")
        grad_map_param_to_grad[W0_vid] = dW0_vid

        if b0_vid is not None:
            db0_vid = pick_next_grad(tuple(v(b0_vid).shape), like_vid=b0_vid, tag="_b0")
            grad_map_param_to_grad[b0_vid] = db0_vid
        else:
            db0_vid = None

        dW1_vid = pick_next_grad(tuple(v(W1_vid).shape), like_vid=W1_vid, tag="_W1")
        grad_map_param_to_grad[W1_vid] = dW1_vid

        if b1_vid is not None:
            db1_vid = pick_next_grad(tuple(v(b1_vid).shape), like_vid=b1_vid, tag="_b1")
            grad_map_param_to_grad[b1_vid] = db1_vid
        else:
            db1_vid = None

        # grads for activations
        d_relu_out_vid = pick_next_grad(tuple(v(relu_out_vid).shape), like_vid=relu_out_vid, tag="_relu")
        d_lin0_out_vid = pick_next_grad(tuple(v(lin0_out_vid).shape), like_vid=lin0_out_vid, tag="_lin0out")
        dx0_vid        = pick_next_grad(tuple(v(x_vid).shape), like_vid=x_vid, tag="_x")

        # ---- Linear1 backward ----
        lowered.append({
            "op": "gemm",
            "attrs": {"transA": False, "transB": False},
            "inputs": [out_grad_vid, W1_vid],
            "outputs": [d_relu_out_vid],
        })
        # dW1 = relu_out^T @ out_grad
        lowered.append({
            "op": "gemm",
            "attrs": {"transA": True, "transB": False},
            "inputs": [relu_out_vid, out_grad_vid],
            "outputs": [dW1_vid],
        })
        if db1_vid is not None:
            lowered.append({
                "op": "reduce_sum",
                "attrs": {"axis": 0, "keepdim": False},
                "inputs": [out_grad_vid],
                "outputs": [db1_vid],
            })

        # ---- ReLU backward ----
        relu_y_saved_vid = None
        for vid, val in ir.values.items():
            if getattr(val, "name", "") == "relu_y_saved" and tuple(val.shape) == tuple(v(relu_out_vid).shape):
                relu_y_saved_vid = int(vid)
                break
        if relu_y_saved_vid is None:
            saved = ir.new_value(
                name="relu_y_saved",
                shape=tuple(v(relu_out_vid).shape),
                dtype=str(v(relu_out_vid).dtype),
                device=str(v(relu_out_vid).device),
            )
            relu_y_saved_vid = int(saved.id)

        # semantic saved copy: op name must be "copy" (trace also "copy")
        lowered.append({"op": "copy", "attrs": {}, "inputs": [relu_out_vid], "outputs": [relu_y_saved_vid]})

        lowered.append({
            "op": "relu_bwd",
            "attrs": {},
            "inputs": [d_relu_out_vid, relu_y_saved_vid],
            "outputs": [d_lin0_out_vid],
        })

        # ---- Linear0 backward ----
        lowered.append({
            "op": "gemm",
            "attrs": {"transA": False, "transB": False},
            "inputs": [d_lin0_out_vid, W0_vid],
            "outputs": [dx0_vid],
        })
        # dW0 = x^T @ d_lin0_out
        lowered.append({
            "op": "gemm",
            "attrs": {"transA": True, "transB": False},
            "inputs": [x_vid, d_lin0_out_vid],
            "outputs": [dW0_vid],
        })
        if db0_vid is not None:
            lowered.append({
                "op": "reduce_sum",
                "attrs": {"axis": 0, "keepdim": False},
                "inputs": [d_lin0_out_vid],
                "outputs": [db0_vid],
            })

    # -------------------------
    # 3) Optim lowering LAST
    # -------------------------
    for n in step_nodes:
        si = int(n.inputs[0])
        so = int(n.outputs[0])
        lowered.append({"op": "step_inc", "attrs": {}, "inputs": [si], "outputs": [so]})

    for n in bc_nodes:
        step_vid = int(n.inputs[0])
        b1_vid = int(n.outputs[0])
        b2_vid = int(n.outputs[1])
        lowered.append({
            "op": "bias_corr",
            "attrs": {"beta1": float(n.attrs["beta1"]), "beta2": float(n.attrs["beta2"])} ,
            "inputs": [step_vid],
            "outputs": [b1_vid, b2_vid],
        })

    warmup_mode = os.getenv("AICF_WARMUP", "0") == "1"

    for n in adam_nodes:
        p_in = int(n.inputs[0])

        # MUST use backward-produced grad if available
        if p_in in grad_map_param_to_grad:
            g_in = int(grad_map_param_to_grad[p_in])
        else:
            # fallback: still never a leaf param.grad vid
            g_in = pick_next_grad(tuple(v(p_in).shape), like_vid=p_in, tag="_fallback")

        m_in = int(n.inputs[2])
        v_in = int(n.inputs[3])
        bc1 = int(n.inputs[4])
        bc2 = int(n.inputs[5])

        # in-place
        p_out = p_in
        m_out = m_in
        v_out = v_in

        lr = float(n.attrs["lr"])
        if warmup_mode:
            lr = 0.0

        lowered.append({
            "op": "adam_step",
            "attrs": {
                "lr": lr,
                "beta1": float(n.attrs["beta1"]),
                "beta2": float(n.attrs["beta2"]),
                "eps": float(n.attrs["eps"]),
            },
            "inputs": [p_in, g_in, m_in, v_in, bc1, bc2],
            "outputs": [p_out, m_out, v_out],
        })

        if os.getenv("AICF_LOWER_ADAM_DEBUG", "0") == "1":
            print(f"[lower][adam] p={p_in} g={g_in} m={m_in} v={v_in} bc1={bc1} bc2={bc2} lr={lr}")

    return lowered


# ============================================================
# Env auto-binding for IRExecutor
# ============================================================

def _iter_vids_from_lowered(lowered: List[Dict[str, Any]]) -> Iterable[int]:
    for it in lowered:
        for x in it.get("inputs", []):
            yield int(x)
        for y in it.get("outputs", []):
            yield int(y)


def autobind_env_from_lowered(
    ir: IRGraph,
    lowered: List[Dict[str, Any]],
    env: Optional[Dict[int, torch.Tensor]] = None,
    *,
    device: Optional[torch.device] = None,
) -> Dict[int, torch.Tensor]:
    """
    Ensure env has runtime tensors for every vid referenced by lowered.
    This binds grad pool values / relu_y_saved / any temporaries created during lowering.
    """
    if env is None:
        env = {}

    # pick default device
    if device is None:
        dev_str = None
        for _, val in ir.values.items():
            dev_str = str(getattr(val, "device", ""))
            if dev_str:
                break
        if dev_str:
            try:
                device = torch.device(dev_str)
            except Exception:
                device = torch.device("cuda")
        else:
            device = torch.device("cuda")

    needed = set(_iter_vids_from_lowered(lowered))

    for vid in sorted(needed):
        if vid in env:
            continue
        val = ir.values.get(int(vid))
        if val is None:
            raise RuntimeError(f"autobind_env_from_lowered: vid {vid} not found in ir.values")

        shape = tuple(getattr(val, "shape"))
        dtype_s = str(getattr(val, "dtype"))

        if "float16" in dtype_s:
            dtype = torch.float16
        elif "bfloat16" in dtype_s:
            dtype = torch.bfloat16
        elif "float32" in dtype_s:
            dtype = torch.float32
        elif "float64" in dtype_s:
            dtype = torch.float64
        elif "int64" in dtype_s:
            dtype = torch.int64
        elif "int32" in dtype_s:
            dtype = torch.int32
        else:
            dtype = torch.float32

        env[vid] = torch.empty(shape, device=device, dtype=dtype)

    return env


# ============================================================
# compile + capture
# ============================================================

def compile_and_capture(
    step_fn,
    *,
    name: str = "train_step",
    warmup_runs: int = 2,
    warmup_sync: bool = True,
    validate: bool = True,
    trace: bool = True,
    enforce_ops: Sequence[str] = ("adam_step",),
    torch_sync: bool = True,
    # accepted for compatibility but unused here
    bind_model: Optional[Any] = None,
    bind_optim: Optional[Any] = None,
    bind_x: Optional[Any] = None,
    bind_t: Optional[Any] = None,
    # new: build env for IRExecutor automatically (recommended)
    autobind_env: bool = True,
) -> CompileArtifact:
    """
    compile -> (optional validate) -> lower -> warmup -> capture -> trace

    IMPORTANT:
      - Warmup MUST NOT update params/states; we set AICF_WARMUP=1 during warmup.
      - Capture MUST run with updates enabled (AICF_WARMUP=0).
      - Capture is performed exactly once (no double-capture).
    """
    ir = compile_ir(step_fn, name=name)
    lowered = lower_to_backend_ops(ir)

    if validate:
        _assert_lowering_has_semantic_saved_copy(ir, lowered)

    # ---- warmup (no state drift) ----
    prev_warm = os.getenv("AICF_WARMUP", None)
    if warmup_runs and warmup_runs > 0:
        os.environ["AICF_WARMUP"] = "1"
        try:
            warmup_capture_safe(train_step=step_fn, runs=warmup_runs, sync=warmup_sync)
        finally:
            if prev_warm is None:
                os.environ.pop("AICF_WARMUP", None)
            else:
                os.environ["AICF_WARMUP"] = prev_warm

    backend = get_backend()

    # reset capture/trace state
    backend.capture_reset()
    if torch_sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    backend.trace_reset()
    backend.trace_enable(bool(trace))

    # ---- single capture ----
    prev_cap = os.getenv("AICF_WARMUP", None)
    os.environ["AICF_WARMUP"] = "0"
    try:
        backend.capture_begin()
        step_fn()
        backend.capture_end()
    finally:
        if prev_cap is None:
            os.environ.pop("AICF_WARMUP", None)
        else:
            os.environ["AICF_WARMUP"] = prev_cap

    if torch_sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    trace_ops: List[str] = backend.trace_get() if trace else []
    backend.trace_enable(False)

    art = CompileArtifact(name=name, ir=ir, lowered=lowered, trace_ops=trace_ops, backend=backend)

    for op in enforce_ops:
        art.assert_trace_has(op)

    if autobind_env:
        env = autobind_env_from_lowered(ir, lowered, env=art.env)
        art.attach_env(env, merge=False)

    return art


__all__ = [
    "CompileArtifact",
    "compile_ir",
    "lower_to_backend_ops",
    "compile_and_capture",
    "autobind_env_from_lowered",
    # tracing exports
    "tracing",
    "is_tracing",
    "get_ir",
    "as_ir_value_obj",
    "as_ir_value_torch",
    "trace_reset_cache",
]
