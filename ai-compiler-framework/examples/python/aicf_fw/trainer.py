# examples/python/aicf_fw/trainer.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Tuple, Any, Optional, Iterator

import torch

from .backend import get_backend
from .modules.base import Module
from .optim.base import Optimizer
from .losses.base import Loss


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

    # -----------------------
    # Step decomposition
    # -----------------------
    def _zero_grad(self) -> None:
        if hasattr(self.optimizer, "zero_grad"):
            try:
                self.optimizer.zero_grad(self.model.parameters())
                return
            except TypeError:
                self.optimizer.zero_grad()
                return

        with torch.no_grad():
            for p in self.model.parameters():
                if hasattr(p, "data") and hasattr(p.data, "t"):
                    if p.data.t.grad is not None:
                        p.data.t.grad = None

    def _forward_loss(self, batch) -> Any:
        x, t = batch
        y = self.model(x)
        loss = self.loss_fn(y, t)
        return loss

    def _backward_opt(self, loss) -> None:
        loss.t.backward()
        self.optimizer.step(self.model.parameters())

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
                    # 1) backend가 고정 입력 버퍼를 제공하면 반드시 그걸로 캡처
                    if hasattr(backend, "prepare_capture_batch"):
                        cap_batch = backend.prepare_capture_batch(batch)
                    else:
                        # prepare_capture_batch가 없으면 capture는 의미가 없음(포인터 불일치)
                        raise RuntimeError(
                            "capture mode requires backend.prepare_capture_batch(batch) "
                            "to create fixed input buffers for CUDA Graph."
                        )

                    if hasattr(backend, "warmup"):
                        backend.warmup(self.model, cap_batch)

                    # 2) 캡처
                    backend.capture_begin()
                    loss = self.train_step_eager(cap_batch)  # ✅ fixed buffers used here
                    backend.capture_end()

                    # 3) replay 후 loss를 읽기 위한 참조 저장
                    self._captured_loss_ref = loss
                    captured = True

                else:
                    # 4) 매 step: 데이터만 copy_로 갱신 후 replay
                    self._backend_bind_batch(backend, batch)
                    _ = backend.replay()

                    # replay가 값을 써넣는 텐서를 읽어옴
                    loss = self._captured_loss_ref

                # 5) optional validation: replay loss vs eager fwd loss
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

        self.cfg.sync_each_step = bool(int(os.environ.get("AICF_SYNC_EACH_STEP", "1" if self.cfg.sync_each_step else "0")))

        validate_every_env = os.environ.get("AICF_VALIDATE_EVERY", None)
        if validate_every_env is not None:
            self.cfg.validate_every = int(validate_every_env)

        capture_at_env = os.environ.get("AICF_CAPTURE_AT_STEP", None)
        if capture_at_env is not None:
            self.cfg.capture_at_step = int(capture_at_env)
