# aicf_fw/core/runtime.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import os
import torch

from aicf_fw.backend import get_backend
from aicf_fw.core.ir import IRGraph


# ============================================================
# Warmup helper (capture-safe)
# ============================================================

@torch.no_grad()
def warmup_capture_safe(
    *,
    train_step: Callable[[], None],
    runs: int = 1,
    sync: bool = True,
) -> None:
    """
    Capture 전에 모든 '포인터가 고정돼야 하는' 버퍼를 미리 materialize.
    - BufferPool/activation 등 런타임 할당
    - leaf parameter.grad 할당
    - 커널 lazy init(첫 호출 때 생성되는 것들) 방출

    IMPORTANT:
      - warmup 중에는 상태 드리프트를 막기 위해 AICF_WARMUP=1 을 기본으로 켬.
        (compile.py에서 이미 켜준 경우도 있으니, 기존 값은 보존)
    """
    if runs <= 0:
        return

    from aicf_fw.core.autograd import in_capture
    if in_capture():
        raise RuntimeError("warmup_capture_safe() must be called OUTSIDE capture.")

    # ensure warmup mode ON unless caller already set it
    prev = os.getenv("AICF_WARMUP", None)
    if prev is None:
        os.environ["AICF_WARMUP"] = "1"

    try:
        for _ in range(runs):
            train_step()
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        # restore env var
        if prev is None:
            os.environ.pop("AICF_WARMUP", None)
        else:
            os.environ["AICF_WARMUP"] = prev


# ============================================================
# IRExecutor debug utilities (diverge tracing)
# ============================================================

def _dbg_on() -> bool:
    return os.getenv("AICF_IREXEC_DIVERGE_DEBUG", "0") == "1"


def _dbg_limit() -> int:
    try:
        return int(os.getenv("AICF_IREXEC_DIVERGE_LIMIT", "50"))
    except Exception:
        return 50


def _is_capturing() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _parse_watch_vids() -> List[int]:
    s = os.getenv("AICF_IREXEC_WATCH_VIDS", "").strip()
    if not s:
        return []
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception:
            pass
    return out


def _gemm_ptr_debug_on() -> bool:
    return os.getenv("AICF_IREXEC_GEMM_PTR_DEBUG", "0") == "1"


def _adam_ptr_debug_on() -> bool:
    # 통일: IRExecutor 쪽 디버그는 IREXEC prefix로
    return os.getenv("AICF_IREXEC_ADAM_PTR_DEBUG", "0") == "1"


def _adam_safe_mode_on() -> bool:
    # 기본값 1(안전모드 ON): mismatch 잡고 나서 0으로 끄면 됨
    return os.getenv("AICF_IREXEC_ADAM_SAFE", "1") == "1"


def _ptr(t: torch.Tensor) -> int:
    try:
        return int(t.data_ptr())
    except Exception:
        return -1


@torch.no_grad()
def _checksum_u32(t: torch.Tensor) -> int:
    """
    Pure-torch "value checksum" for divergence hunting.
    - capture 중에는 값 접근 금지 → -1
    - float는 quantize 후 int64로 누적합
    """
    if _is_capturing():
        return -1

    tt = t.detach()
    if tt.numel() == 0:
        return 0

    flat = tt.reshape(-1)
    if flat.dtype.is_floating_point:
        q = torch.round(flat.float() * 1e6).to(torch.int64)
    else:
        q = flat.to(torch.int64)

    idx = torch.arange(1, q.numel() + 1, device=q.device, dtype=torch.int64)
    s = (q * idx).sum()
    return int((s & 0xFFFFFFFF).item())


@torch.no_grad()
def _tensor_sig_tuple(t: torch.Tensor) -> tuple:
    """
    반환:
      (shape, stride, storage_offset, contig, dtype(str), device(str),
       mean, maxabs, norm, chk, ptr)
    capture 중이면 값 통계는 None, chk=-1
    """
    tt = t.detach()
    shape = tuple(tt.shape)
    stride = tuple(tt.stride())
    storage_offset = int(tt.storage_offset())
    contig = bool(tt.is_contiguous())
    dtype = str(tt.dtype)
    dev = str(tt.device)
    ptr = int(tt.data_ptr())

    if _is_capturing() or tt.numel() == 0:
        return (shape, stride, storage_offset, contig, dtype, dev, None, None, None, -1, ptr)

    tf = tt.float()
    mean = float(tf.mean().item())
    maxabs = float(tf.abs().max().item())
    norm = float(tf.norm().item())
    chk = _checksum_u32(tt)
    return (shape, stride, storage_offset, contig, dtype, dev, mean, maxabs, norm, chk, ptr)


# ============================================================
# IRExecutor
# ============================================================

@dataclass
class IRExecutor:
    """
    IR-only executor:
      - Executes lowered ops list (artifact.lowered)
      - Uses env: vid(int) -> torch.Tensor
      - Dispatch via backend.op_call_out(op, inputs_t, outputs_t, attrs)
    """
    ir: IRGraph
    lowered: List[Dict[str, Any]]
    env: Dict[int, torch.Tensor]
    backend: Any = None

    # debug storage
    _last_intermediates: Dict[str, torch.Tensor] | None = None
    _last_sigs: Dict[str, Any] | None = None

    @staticmethod
    def from_artifact(art: Any) -> "IRExecutor":
        if not hasattr(art, "runtime_env"):
            raise RuntimeError("IRExecutor.from_artifact: artifact has no runtime_env()")
        env = art.runtime_env()
        if not isinstance(env, dict) or len(env) == 0:
            aname = getattr(art, "name", "<unnamed>")
            raise RuntimeError(
                f"IRExecutor.from_artifact: empty env for artifact={aname}. "
                "Attach env first: art.attach_env(vid->torch.Tensor)."
            )
        return IRExecutor(ir=art.ir, lowered=art.lowered, env=dict(env), backend=getattr(art, "backend", None))

    def _get_t(self, vid: int, *, op: str = "?", io: str = "?") -> torch.Tensor:
        t = self.env.get(int(vid), None)
        if t is None:
            v = self.ir.values.get(int(vid), None)
            meta = ""
            if v is not None:
                meta = f" name={v.name} shape={tuple(v.shape)} dtype={v.dtype} device={v.device}"
            raise RuntimeError(f"IRExecutor: missing runtime tensor for vid={vid} ({io} of {op}).{meta}")
        return t

    def get_last_intermediates(self) -> Dict[str, torch.Tensor]:
        return dict(self._last_intermediates or {})

    def get_last_sigs(self) -> Dict[str, Any]:
        return dict(self._last_sigs or {})

    def _vname(self, vid: int) -> str:
        v = self.ir.values.get(int(vid), None)
        return v.name if v is not None else "<?>"

    def _dispatch_op(self, bk, op: str, inputs_t: List[torch.Tensor], outputs_t: List[torch.Tensor], attrs: Dict[str, Any]):
        """
        Compatibility shim:
          - lowering에서 copy_saved/copy_aux로 분리했더라도
            backend가 copy_saved를 모르면 여기서 copy로 매핑.
        """
        if op in ("copy_saved", "copy_aux"):
            bk.op_call_out("copy", inputs_t, outputs_t, attrs)
            return
        bk.op_call_out(op, inputs_t, outputs_t, attrs)

    @torch.no_grad()
    def run(
        self,
        *,
        debug_nan: bool = False,
        debug_intermediate: bool = False,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor] | None:
        bk = self.backend or get_backend()

        watch_vids = _parse_watch_vids()
        watch_set = set(int(v) for v in watch_vids)
        gemm_ptr_dbg = _gemm_ptr_debug_on()
        adam_ptr_dbg = _adam_ptr_debug_on()
        adam_safe = _adam_safe_mode_on()

        def _t_meta(t: torch.Tensor) -> str:
            return f"shape={tuple(t.shape)} dtype={t.dtype} dev={t.device}"

        def _nonfinite_count(t: torch.Tensor) -> tuple[int, int]:
            if _is_capturing():
                return (0, 0)
            return int(torch.isnan(t).sum().item()), int(torch.isinf(t).sum().item())

        def _watch_dump(prefix: str, idx: int, op: str):
            if not watch_set:
                return
            for wv in sorted(list(watch_set)):
                if wv not in self.env:
                    continue
                wt = self.env[wv]
                chk = _checksum_u32(wt)
                chk_s = f"{chk}" if chk >= 0 else "skipped"
                print(
                    f"[irexec][watch]{prefix} op#{idx:03d} {op:10s} "
                    f"vid{wv}({self._vname(wv)}) chk={chk_s} ptr={int(wt.data_ptr())} shape={tuple(wt.shape)}"
                )

        # reset debug storages per-run
        if debug_intermediate or return_intermediates:
            self._last_intermediates = {}
            self._last_sigs = {}
        else:
            self._last_intermediates = None
            self._last_sigs = None

        # optional: print canonical scalars once
        if _dbg_on():
            for vid, tag in [(12, "step"), (14, "bc1_inv"), (15, "bc2_inv")]:
                if int(vid) in self.env and int(vid) in self.ir.values:
                    v = self.ir.values[int(vid)]
                    t = self.env[int(vid)]
                    print(
                        f"[irexec][canon] {tag}: vid={vid} name={v.name} ptr={t.data_ptr()} "
                        f"shape={tuple(t.shape)} dtype={t.dtype} dev={t.device}"
                    )

        dbg_budget = _dbg_limit()
        dbg_count = 0

        for idx, item in enumerate(self.lowered):
            op = item["op"]
            attrs = dict(item.get("attrs", {}))
            in_vids = list(item.get("inputs", []))
            out_vids = list(item.get("outputs", []))

            inputs_t = [self._get_t(v, op=op, io="in") for v in in_vids]
            outputs_t = [self._get_t(v, op=op, io="out") for v in out_vids]

            # watch BEFORE
            _watch_dump(prefix="[before]", idx=idx, op=op)

            # ------------------------------------------------------------
            # adam_step: 안전모드(기본 ON)
            #  - backend 호출은 temp(out-of-place)로 받고
            #  - 끝나고 copy_ 로 원래 p/m/v 갱신 (논리적 in-place)
            # ------------------------------------------------------------
            if op == "adam_step":
                if len(inputs_t) < 6:
                    raise RuntimeError(
                        f"IRExecutor: adam_step expects 6 inputs [p,g,m,v,bc1,bc2], got {len(inputs_t)}"
                    )
                if len(outputs_t) < 3:
                    raise RuntimeError(
                        f"IRExecutor: adam_step expects 3 outputs [p,m,v], got {len(outputs_t)}"
                    )

                if adam_ptr_dbg:
                    in_ptrs = [_ptr(t) for t in inputs_t[:6]]
                    out_ptrs = [_ptr(t) for t in outputs_t[:3]]
                    print(
                        f"[irexec][adam#{idx:03d}] "
                        f"in_vids={in_vids} out_vids={out_vids} "
                        f"in_ptrs={in_ptrs} out_ptrs={out_ptrs} safe={int(adam_safe)}"
                    )

                if debug_intermediate or return_intermediates:
                    names = ["p", "g", "m", "v", "bc1", "bc2"]
                    for j in range(6):
                        t = inputs_t[j]
                        key = f"op#{idx:03d}:{op}:adam_in{j}:{names[j]}"
                        self._last_intermediates[key] = t
                        self._last_sigs[key] = _tensor_sig_tuple(t)

                if adam_safe:
                    p_tgt, m_tgt, v_tgt = inputs_t[0], inputs_t[2], inputs_t[3]
                    p_tmp = torch.empty_like(p_tgt)
                    m_tmp = torch.empty_like(m_tgt)
                    v_tmp = torch.empty_like(v_tgt)
                    outputs_t = [p_tmp, m_tmp, v_tmp]
                else:
                    outputs_t = [inputs_t[0], inputs_t[2], inputs_t[3]]

            # capture intermediates BEFORE call
            if debug_intermediate or return_intermediates:
                for j, (v, t) in enumerate(zip(in_vids, inputs_t)):
                    nm = self._vname(int(v))
                    key = f"op#{idx:03d}:{op}:in{j}:vid{int(v)}:{nm}"
                    self._last_intermediates[key] = t
                    self._last_sigs[key] = _tensor_sig_tuple(t)

                for j, (v, t) in enumerate(zip(out_vids, outputs_t)):
                    nm = self._vname(int(v))
                    key = f"op#{idx:03d}:{op}:out{j}:vid{int(v)}:{nm}:BEFORE"
                    self._last_intermediates[key] = t
                    self._last_sigs[key] = _tensor_sig_tuple(t)

            # verbose per-op IO when debug on
            if _dbg_on() and dbg_count < dbg_budget:
                dbg_count += 1
                print(f"[irexec][op#{idx:03d}] {op} attrs={attrs}")
                for v, t in zip(in_vids, inputs_t):
                    print(f"  in  vid={int(v):3d} name={self._vname(int(v)):12s} {_t_meta(t)} ptr={t.data_ptr()}")
                for v, t in zip(out_vids, outputs_t):
                    print(f"  out vid={int(v):3d} name={self._vname(int(v)):12s} {_t_meta(t)} ptr={t.data_ptr()}")

            # extra gemm ptr debug
            if gemm_ptr_dbg and op == "gemm":
                in_ptrs = [int(t.data_ptr()) for t in inputs_t]
                out_ptrs = [int(t.data_ptr()) for t in outputs_t]
                in_names = [f"{int(v)}({self._vname(int(v))})" for v in in_vids]
                out_names = [f"{int(v)}({self._vname(int(v))})" for v in out_vids]
                watch_ptrs = {wv: int(self.env[wv].data_ptr()) for wv in watch_set if wv in self.env}
                print(
                    f"[irexec][gemm#{idx:03d}] "
                    f"in_vids={in_vids} in_names={in_names} out_vids={out_vids} out_names={out_names} "
                    f"in_ptrs={in_ptrs} out_ptrs={out_ptrs} watch_ptrs={watch_ptrs}"
                )

            # dispatch (with compatibility mapping)
            self._dispatch_op(bk, op, inputs_t, outputs_t, attrs)

            # if adam_safe: copy back into original targets (logical in-place)
            if op == "adam_step" and adam_safe:
                p_tgt, m_tgt, v_tgt = inputs_t[0], inputs_t[2], inputs_t[3]
                p_tgt.copy_(outputs_t[0])
                m_tgt.copy_(outputs_t[1])
                v_tgt.copy_(outputs_t[2])
                outputs_t = [p_tgt, m_tgt, v_tgt]

            # SSA rebind
            for ov, ot in zip(out_vids, outputs_t):
                self.env[int(ov)] = ot

            # capture intermediates AFTER call
            if debug_intermediate or return_intermediates:
                for j, (v, t) in enumerate(zip(out_vids, outputs_t)):
                    nm = self._vname(int(v))
                    key = f"op#{idx:03d}:{op}:out{j}:vid{int(v)}:{nm}:AFTER"
                    self._last_intermediates[key] = t
                    self._last_sigs[key] = _tensor_sig_tuple(t)

            # watch AFTER
            _watch_dump(prefix="[after ]", idx=idx, op=op)

            # after-op compact line
            if _dbg_on() and dbg_count <= dbg_budget and len(outputs_t) > 0:
                ov0 = int(out_vids[0])
                ot0 = outputs_t[0]
                nm0 = self._vname(ov0)
                chk = _checksum_u32(ot0)
                chk_s = f"{chk}" if chk >= 0 else "skipped"
                if _is_capturing():
                    print(f"[irexec][after#{idx:03d}] {op} -> {nm0}(vid={ov0}) ptr={ot0.data_ptr()} {_t_meta(ot0)} chk={chk_s}")
                else:
                    tf = ot0.detach()
                    if tf.numel() == 0:
                        print(f"[irexec][after#{idx:03d}] {op} -> {nm0}(vid={ov0}) ptr={ot0.data_ptr()} (empty) chk={chk_s}")
                    else:
                        t32 = tf.float()
                        mean = float(t32.mean().item())
                        maxabs = float(t32.abs().max().item())
                        nrm = float(t32.norm().item())
                        print(f"[irexec][after#{idx:03d}] {op} -> {nm0}(vid={ov0}) ptr={ot0.data_ptr()} mean={mean:+.3e} maxabs={maxabs:.3e} norm={nrm:.3e} chk={chk_s}")

            # nan/inf guard
            if debug_nan and len(outputs_t) > 0 and not _is_capturing():
                for j, (ov, ot) in enumerate(zip(out_vids, outputs_t)):
                    nn, ni = _nonfinite_count(ot)
                    if nn or ni:
                        ins_meta = ", ".join([f"{iv}:{_t_meta(it)}" for iv, it in zip(in_vids, inputs_t)])
                        outs_meta = ", ".join([f"{ov}:{_t_meta(ot)}" for ov, ot in zip(out_vids, outputs_t)])
                        raise RuntimeError(
                            f"IRExecutor: first non-finite after op#{idx:02d} '{op}' out#{j} vid={int(ov)} "
                            f"(nan={nn}, inf={ni}).\n"
                            f"  inputs: {ins_meta}\n"
                            f"  outputs: {outs_meta}\n"
                            f"  attrs: {attrs}"
                        )

        if return_intermediates:
            return dict(self._last_intermediates or {})
        return None
