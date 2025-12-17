# examples/python/aicf_fw/trainer.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Tuple, Any, Optional

import torch

from .backend import get_backend
from .modules.base import Module
from .optim.base import Optimizer
from .losses.base import Loss


@dataclass
class TrainerConfig:
    mode: str = "eager"      # "eager" | "bench" | "capture"
    log_every: int = 10


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
        get_backend().set_mode(self.cfg.mode)

    def train_step(self, batch):
        x, t = batch

        # ✅ grad 초기화 (누적 방지) - leaf 텐서 인플레이스 문제 회피 포함
        # NOTE: p.data.t.grad = None 이 가장 깔끔함 (zero_도 가능)
        with torch.no_grad():
            for p in self.model.parameters():
                if p.data.t.grad is not None:
                    p.data.t.grad = None

        y = self.model(x)
        loss = self.loss_fn(y, t)

        # backward
        loss.t.backward()

        # (디버그용) 첫 파라미터 grad 확인
        if int(os.environ.get("AICF_DEBUG_GRAD", "0")) == 1:
            p0 = next(iter(self.model.parameters()))
            g = p0.data.t.grad
            print(
                "grad none?", g is None,
                "grad mean:", (g.abs().mean().item() if g is not None else None)
            )

        # optimizer step (Optimizer 내부에서 torch.no_grad() 처리 권장)
        self.optimizer.step(self.model.parameters())
        return loss

    def fit(self, dataloader: Iterable[Tuple[Any, Any]], steps: int):
        backend = get_backend()

        # --- profiling controls (env) ---
        # AICF_WARMUP_STEPS: 이 step까지는 그냥 굴려서 안정화
        # AICF_PROFILE_STEP: 특정 step에서만 동기화로 커널을 "한 덩어리"로 모음
        # AICF_SYNC_EACH_STEP: 디버깅용 강제 동기화
        warmup_steps = int(os.environ.get("AICF_WARMUP_STEPS", "0"))
        profile_step_env = os.environ.get("AICF_PROFILE_STEP", None)
        profile_step = int(profile_step_env) if profile_step_env is not None else None
        sync_each = int(os.environ.get("AICF_SYNC_EACH_STEP", "0"))

        # --- capture mode ---
        if self.cfg.mode == "capture":
            sample = next(iter(dataloader))
            backend.warmup(self.model, sample)
            backend.capture_begin()
            self.train_step(sample)  # 캡처 대상
            backend.capture_end()

        for step, batch in enumerate(dataloader):
            if step >= steps:
                break

            # warmup 구간: 그대로 train_step 수행 후 continue (커널 캐시/알고리즘 선택 안정화)
            if step < warmup_steps:
                if self.cfg.mode != "capture":
                    _ = self.train_step(batch)
                else:
                    _ = backend.replay() or None
                continue

            do_profile = (profile_step is not None and step == profile_step)

            if do_profile and torch.cuda.is_available():
                torch.cuda.synchronize()

            loss = self.train_step(batch) if self.cfg.mode != "capture" else (backend.replay() or None)

            if do_profile and torch.cuda.is_available():
                torch.cuda.synchronize()

            if sync_each and torch.cuda.is_available():
                torch.cuda.synchronize()

            if (step % self.cfg.log_every) == 0:
                # loss가 Tensor 래퍼일 수도 / None 일 수도 있으니 안전하게 처리
                if loss is None:
                    print(f"[step {step}] loss=None")
                else:
                    # Tensor 래퍼면 .t.item(), torch tensor면 .item()
                    val = loss.t.item() if hasattr(loss, "t") else loss.item()
                    print(f"[step {step}] loss={val:.4f}")

                # bench 모드면 op breakdown 같은 것도 출력 가능
                if hasattr(backend, "profiler") and backend.profiler is not None:
                    backend.profiler.maybe_report()
