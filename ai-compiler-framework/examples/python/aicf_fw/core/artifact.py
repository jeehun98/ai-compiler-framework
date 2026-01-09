# aicf_fw/core/artifact.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from aicf_fw.core.train_state import TrainState


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


@torch.no_grad()
def _snapshot_params(model: Any) -> Dict[str, torch.Tensor]:
    return {n: p.data.detach().clone() for n, p in model.named_parameters()}


@torch.no_grad()
def _max_param_diff(model: Any, snaps: Dict[str, torch.Tensor]) -> float:
    m = 0.0
    for n, p in model.named_parameters():
        d = (p.data - snaps[n]).abs().max().item()
        if d > m:
            m = float(d)
    return m


@dataclass
class CompileArtifact:
    """
    Compilation artifact produced by compile/lower/capture.

    Fields:
      - ir: compiled IR object (must have dump_json if you want printing)
      - lowered: list of backend ops (dicts with at least "op")
      - trace_ops: runtime trace op names (strings) captured during CUDA graph capture
      - backend: backend instance (must support replay())
    """
    name: str
    ir: Any
    lowered: List[Dict[str, Any]]
    trace_ops: List[str]
    backend: Any

    def assert_trace_has(self, op: str) -> None:
        if op not in self.trace_ops:
            raise AssertionError(f"[trace] {op} missing in captured runtime ops: {self.trace_ops}")

    def assert_runtime_matches_lowering(
        self,
        model: Any,
        *,
        trace_filter: bool = True,
        ignore_ops: Optional[Sequence[str]] = None,
        opt_ops: Sequence[str] = ("step_inc", "bias_corr", "adam_step"),
    ) -> None:
        """
        Ported from PR3: compare lowered ops vs runtime trace.
        - If trace_filter=True: ignore some noisy ops + compare forward-slice and optim-slice.
        - Else: strict full sequence match.
        """
        trace_ops = list(self.trace_ops)
        lower_ops = [x["op"] for x in self.lowered]

        if not trace_filter:
            if trace_ops != lower_ops:
                raise AssertionError(
                    "Lowering mismatch (strict):\n"
                    f"  trace_ops={trace_ops}\n"
                    f"  lower_ops={lower_ops}\n"
                )
            return

        IGNORE = set(ignore_ops) if ignore_ops is not None else {"grad_zero", "copy", "relu_bwd", "reduce_sum"}
        tr = [op for op in trace_ops if op not in IGNORE]
        lo = list(lower_ops)

        # forward slice until mse_grad (inclusive)
        tr_fw = tr[: tr.index("mse_grad") + 1] if "mse_grad" in tr else tr
        lo_fw = lo[: lo.index("mse_grad") + 1] if "mse_grad" in lo else lo
        if tr_fw != lo_fw:
            raise AssertionError(
                "Lowering mismatch (forward-slice):\n"
                f"  trace_fw={tr_fw}\n"
                f"  lower_fw={lo_fw}\n"
            )

        # optim slice
        KEEP_OPT = set(opt_ops)
        tr_opt = [op for op in tr if op in KEEP_OPT]
        lo_opt = [op for op in lo if op in KEEP_OPT]

        N_PARAM = len(list(model.named_parameters()))
        exp = ["step_inc", "bias_corr"] + ["adam_step"] * N_PARAM

        if tr_opt != exp:
            raise AssertionError(
                "Runtime trace optim-slice mismatch:\n"
                f"  trace_opt={tr_opt}\n"
                f"  expected={exp}\n"
                f"  N_PARAM={N_PARAM}\n"
            )
        if lo_opt != exp:
            raise AssertionError(
                "Lowering optim-slice mismatch:\n"
                f"  lower_opt={lo_opt}\n"
                f"  expected={exp}\n"
                f"  N_PARAM={N_PARAM}\n"
            )

    @torch.no_grad()
    def assert_adam_state_mutates(self, model: Any, optim: Any, *, tag: str = "smoke") -> None:
        """
        Ported from PR3: one replay must advance step and mutate params + m/v.
        """
        st0 = TrainState.capture(model, optim)
        self.backend.replay()
        torch.cuda.synchronize()
        st1 = TrainState.capture(model, optim)

        # step must advance
        if int(st1.step.item()) == int(st0.step.item()):
            raise AssertionError(
                f"[adam][{tag}] step did not advance on replay: {int(st0.step.item())} -> {int(st1.step.item())}"
            )

        # params must change
        max_p = 0.0
        for n in st0.params.keys():
            max_p = max(max_p, _max_abs_diff(st1.params[n], st0.params[n]))
        if max_p == 0.0:
            raise AssertionError(
                f"[adam][{tag}] params did not change on replay. "
                f"Possible causes: (1) gradients are zero/None, (2) adam_step kernel is no-op, "
                f"(3) captured step uses uninitialized static inputs. "
            )

        # m/v must change
        max_m = 0.0
        max_v = 0.0
        for i in st0.adam_m.keys():
            max_m = max(max_m, _max_abs_diff(st1.adam_m[i], st0.adam_m[i]))
            max_v = max(max_v, _max_abs_diff(st1.adam_v[i], st0.adam_v[i]))
        if max_m == 0.0 or max_v == 0.0:
            raise AssertionError(
                f"[adam][{tag}] m/v did not change on replay (expected adam_step mutation) "
                f"(max_m={max_m}, max_v={max_v})"
            )

    @torch.no_grad()
    def assert_determinism(
        self,
        model: Any,
        optim: Any,
        *,
        replays: int = 20,
        check_restore: bool = True,
        print_every: int = 0,
        loss_fn: Optional[Callable[[], float]] = None,
        tag: str = "",
    ) -> None:
        """
        Ported from PR3: snapshot full train state st0, run A sequence, restore, run B sequence,
        and compare stepdiff sequences exactly.
        """
        # Snapshot AFTER capture is assumed in caller; but determinism check should use current state.
        st0 = TrainState.capture(model, optim)

        # Optional: verify restore works before any run
        if check_restore:
            st0.restore(model, optim)
            torch.cuda.synchronize()
            st0.assert_equal(model, optim, tag=f"{tag}after-restore(pre-run)")

        # Run A
        stepdiff_A: List[float] = []
        snaps = _snapshot_params(model)
        for i in range(replays):
            self.backend.replay()
            torch.cuda.synchronize()
            sd = _max_param_diff(model, snaps)
            stepdiff_A.append(sd)
            snaps = _snapshot_params(model)
            if print_every and (i % print_every == 0):
                if loss_fn is not None:
                    _ = loss_fn()  # caller decides printing; we keep it side-effect free here

        # Restore
        st0.restore(model, optim)
        torch.cuda.synchronize()
        if check_restore:
            st0.assert_equal(model, optim, tag=f"{tag}after-restore")

        # Run B
        stepdiff_B: List[float] = []
        snaps = _snapshot_params(model)
        for i in range(replays):
            self.backend.replay()
            torch.cuda.synchronize()
            sd = _max_param_diff(model, snaps)
            stepdiff_B.append(sd)
            snaps = _snapshot_params(model)
            if print_every and (i % print_every == 0):
                if loss_fn is not None:
                    _ = loss_fn()

        # Compare exactly (matches PR3 behavior)
        for i, (a, b) in enumerate(zip(stepdiff_A, stepdiff_B)):
            if a != b:
                raise AssertionError(
                    f"Replay determinism broken (stepdiff sequence) at iter {i:02d}: {a:.6e} != {b:.6e}"
                )
