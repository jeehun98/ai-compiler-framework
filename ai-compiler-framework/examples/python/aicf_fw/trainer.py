from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Tuple, Any, Optional, Iterator

import torch

from .backend import get_backend
from .modules.base import Module
from .optim.base import Optimizer
from .losses.base import Loss

# AICF autograd minimal engine
from . import autograd as AG


def _unwrap_t(x: Any) -> torch.Tensor:
    """
    Safe unwrap:
      - aicf_fw.tensor.Tensor -> torch.Tensor via .t
      - torch.Tensor -> itself
    Avoids the 'torch.Tensor.t' method confusion.
    """
    if isinstance(x, torch.Tensor):
        return x
    # our wrapper Tensor has attribute 't' which is torch.Tensor
    if hasattr(x, "t"):
        tt = getattr(x, "t")
        if isinstance(tt, torch.Tensor):
            return tt
    raise TypeError(f"expected aicf_fw.Tensor or torch.Tensor, got {type(x)}")


@dataclass
class TrainerConfig:
    mode: str = "eager"      # "eager" | "bench" | "capture"
    log_every: int = 10

    # orchestration
    warmup_steps: int = 0            # kernel/alloc 안정화 + (실제 학습도 수행)
    capture_at_step: int = -1        # -1 => warmup 직후 바로 캡처
    validate_every: int = 0          # 0 => off. capture replay 중 가끔 eager fwd loss 비교

    # profiling
    profile_step: Optional[int] = None
    sync_each_step: bool = False

    # capture correctness constraints
    strict_capture_requires_bind: bool = True


class Trainer:
    def __init__(
        self,
        model: Module,
        loss_fn: Loss,
        optimizer: Optimizer,
        cfg: Optional[TrainerConfig] = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.cfg = cfg or TrainerConfig()

        self._apply_env_overrides()
        get_backend().set_mode(self.cfg.mode)

        # capture mode: replay 후 loss를 읽기 위한 참조(캡처 때 반환된 loss 텐서)
        self._captured_loss_ref: Optional[Any] = None

        # store last forward (needed for manual backward)
        self._last_y: Optional[Any] = None
        self._last_t: Optional[Any] = None

    # -----------------------
    # Step decomposition
    # -----------------------
    def _zero_grad(self) -> None:
        # prefer optimizer.zero_grad if exists
        if hasattr(self.optimizer, "zero_grad"):
            try:
                self.optimizer.zero_grad(self.model.parameters())
                return
            except TypeError:
                self.optimizer.zero_grad()
                return

        # fallback: set .grad=None on torch tensors
        with torch.no_grad():
            for p in self.model.parameters():
                if hasattr(p, "data") and hasattr(p.data, "t"):
                    tt = p.data.t
                    if isinstance(tt, torch.Tensor) and tt.grad is not None:
                        tt.grad = None

    def _forward_loss(self, batch) -> Any:
        x, t = batch
        AG.tape().clear()
        y = self.model(x)
        self._last_y = y
        self._last_t = t
        loss = self.loss_fn(y, t)
        return loss

    def _backward_opt(self, loss) -> None:
        if os.environ.get("AICF_BACKEND", "torch").lower() != "aicf":
            raise RuntimeError("Manual AICF backward requires AICF_BACKEND=aicf")

        if self._last_y is None or self._last_t is None:
            raise RuntimeError("missing last forward y/t")

        # unwrap y,t safely
        Y = _unwrap_t(self._last_y).contiguous()
        T = _unwrap_t(self._last_t).contiguous()

        # dY from mse grad (kernel default scale = 2/numel unless scale attr used)
        dy = AG.aicf_mse_grad(Y, T)  # torch.Tensor

        # reverse over recorded ops
        for op in reversed(AG.tape().ops):
            if op.kind == "relu":
                # op.x should be torch.Tensor (recommended) or Tensor wrapper
                X_relu = _unwrap_t(op.x).contiguous()
                dy = AG.aicf_relu_bwd(dy.contiguous(), X_relu)

            elif op.kind == "linear":
                # op.x: input activation, op.w_param/op.b_param: parameters
                X = _unwrap_t(op.x).contiguous()          # [B,in]
                dY = dy.contiguous()                      # [B,out]

                W_raw = op.w_param.data.t
                if not isinstance(W_raw, torch.Tensor):
                    raise RuntimeError("linear weight storage must be torch.Tensor")
                W_raw = W_raw.contiguous()

                if W_raw.ndim != 2:
                    raise RuntimeError(f"linear weight must be 2D, got {W_raw.ndim}D")

                B = dY.shape[0]
                in_feat = X.shape[1]
                out_feat = dY.shape[1]

                # Determine original layout: [out,in] vs [in,out]
                if tuple(W_raw.shape) == (out_feat, in_feat):
                    W_used = W_raw
                    w_layout = "out_in"
                elif tuple(W_raw.shape) == (in_feat, out_feat):
                    W_used = W_raw.t().contiguous()
                    w_layout = "in_out"
                else:
                    raise RuntimeError(
                        f"weight shape mismatch: got {tuple(W_raw.shape)}, "
                        f"expected {(out_feat, in_feat)} or {(in_feat, out_feat)}"
                    )

                # dX = dY @ W_used -> [B,in]
                dx = AG.aicf_gemm(dY, W_used, out_shape=(B, in_feat))

                # dW_used = dY^T @ X -> [out,in]
                dW_used = AG.aicf_gemm(dY.t().contiguous(), X, out_shape=(out_feat, in_feat))

                # store grad in ORIGINAL layout
                if w_layout == "out_in":
                    op.w_param.data.t.grad = dW_used
                else:
                    op.w_param.data.t.grad = dW_used.t().contiguous()

                # db = sum over batch -> [out]
                if op.b_param is not None:
                    # ReduceSum kernel currently supports last-dim reduction:
                    # so pass dY^T and axis=-1 -> [out]
                    dY_T = dY.t().contiguous()  # [out,B]
                    db = AG.aicf_reduce_sum(dY.contiguous(), axis=-1, out_shape=(out_feat,))
                    op.b_param.data.t.grad = db

                dy = dx

            else:
                raise RuntimeError(f"unknown op kind: {op}")

        # update (SGDStep or torch fallback depending on optimizer impl)
        self.optimizer.step(self.model.parameters())
        AG.tape().clear()

    def train_step_eager(self, batch) -> Any:
        self._zero_grad()
        loss = self._forward_loss(batch)
        self._backward_opt(loss)
        return loss

    @torch.no_grad()
    def _forward_loss_nograd(self, batch) -> float:
        loss = self._forward_loss(batch)
        if hasattr(loss, "t"):
            return float(loss.t.item())
        return float(loss.item())

    def _loss_to_float(self, loss_like: Any) -> Optional[float]:
        if loss_like is None:
            return None
        if hasattr(loss_like, "t"):
            try:
                return float(loss_like.t.item())
            except Exception:
                return None
        if hasattr(loss_like, "item"):
            try:
                return float(loss_like.item())
            except Exception:
                return None
        try:
            return float(loss_like)
        except Exception:
            return None

    # -----------------------
    # Backend bind shim
    # -----------------------
    def _backend_bind_batch(self, backend, batch) -> None:
        if hasattr(backend, "bind_batch"):
            backend.bind_batch(batch)
            return
        if hasattr(backend, "set_inputs"):
            backend.set_inputs(batch)
            return

        if self.cfg.strict_capture_requires_bind:
            raise RuntimeError(
                "capture mode requires backend.bind_batch(batch) or backend.set_inputs(batch)."
            )

    def _maybe_sync(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _maybe_profile_sync(self, step: int) -> bool:
        return self.cfg.profile_step is not None and step == self.cfg.profile_step

    # -----------------------
    # Main loop
    # -----------------------
    def fit(self, dataloader: Iterable[Tuple[Any, Any]], steps: int) -> None:
        backend = get_backend()
        it: Iterator[Tuple[Any, Any]] = iter(dataloader)

        # -------- Phase 0: warmup (real training) --------
        warmup_n = max(0, int(self.cfg.warmup_steps))
        for step in range(min(warmup_n, steps)):
            batch = next(it)

            do_profile = self._maybe_profile_sync(step)
            if do_profile:
                self._maybe_sync()

            loss = self.train_step_eager(batch)

            if do_profile or self.cfg.sync_each_step:
                self._maybe_sync()

            if (step % self.cfg.log_every) == 0:
                v = self._loss_to_float(loss)
                print(f"[step {step}] loss={v:.4f}" if v is not None else f"[step {step}] loss=None")

        if warmup_n >= steps:
            return

        # -------- Phase 1: capture orchestration --------
        captured = False
        capture_step = self.cfg.capture_at_step
        if capture_step < 0:
            capture_step = warmup_n  # warmup 직후

        current_step = warmup_n

        while current_step < steps:
            batch = next(it)

            do_profile = self._maybe_profile_sync(current_step)
            if do_profile:
                self._maybe_sync()

            # ---- non-capture mode (eager/bench) ----
            if self.cfg.mode != "capture":
                loss = self.train_step_eager(batch)

            # ---- capture mode ----
            else:
                if (not captured) and (current_step == capture_step):
                    if hasattr(backend, "prepare_capture_batch"):
                        cap_batch = backend.prepare_capture_batch(batch)
                    else:
                        raise RuntimeError(
                            "capture mode requires backend.prepare_capture_batch(batch) "
                            "to create fixed input buffers for CUDA Graph."
                        )

                    if hasattr(backend, "warmup"):
                        backend.warmup(self.model, cap_batch)

                    backend.capture_begin()
                    loss = self.train_step_eager(cap_batch)
                    backend.capture_end()

                    self._captured_loss_ref = loss
                    captured = True

                else:
                    self._backend_bind_batch(backend, batch)
                    _ = backend.replay()
                    loss = self._captured_loss_ref

                if self.cfg.validate_every and (current_step % self.cfg.validate_every == 0):
                    eager_loss = self._forward_loss_nograd(batch)
                    replay_loss = self._loss_to_float(self._captured_loss_ref)
                    if replay_loss is not None:
                        diff = abs(replay_loss - eager_loss)
                        if diff > 1e-3:
                            print(
                                f"[validate step {current_step}] "
                                f"replay_loss={replay_loss:.6f} eager_fwd_loss={eager_loss:.6f} diff={diff:.6f}"
                            )

            if do_profile or self.cfg.sync_each_step:
                self._maybe_sync()

            # ---- logging ----
            if (current_step % self.cfg.log_every) == 0:
                v = self._loss_to_float(loss)
                print(f"[step {current_step}] loss={v:.4f}" if v is not None else f"[step {current_step}] loss=None")

                if hasattr(backend, "profiler") and backend.profiler is not None:
                    backend.profiler.maybe_report()

            current_step += 1

    # -----------------------
    # Env override compatibility
    # -----------------------
    def _apply_env_overrides(self) -> None:
        self.cfg.warmup_steps = int(os.environ.get("AICF_WARMUP_STEPS", str(self.cfg.warmup_steps)))

        profile_step_env = os.environ.get("AICF_PROFILE_STEP", None)
        if profile_step_env is not None:
            self.cfg.profile_step = int(profile_step_env)

        self.cfg.sync_each_step = bool(
            int(os.environ.get("AICF_SYNC_EACH_STEP", "1" if self.cfg.sync_each_step else "0"))
        )

        validate_every_env = os.environ.get("AICF_VALIDATE_EVERY", None)
        if validate_every_env is not None:
            self.cfg.validate_every = int(validate_every_env)

        capture_at_env = os.environ.get("AICF_CAPTURE_AT_STEP", None)
        if capture_at_env is not None:
            self.cfg.capture_at_step = int(capture_at_env)
