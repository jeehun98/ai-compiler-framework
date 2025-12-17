# examples/python/aicf_fw/utils/profiling.py
from __future__ import annotations
from contextlib import contextmanager
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class OpStat:
    calls: int = 0
    total_ms: float = 0.0


class OpProfiler:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.stats: Dict[str, OpStat] = {}
        self._last_report_t = time.time()

    @contextmanager
    def scope(self, op_name: str, sig: str, mode: str):
        if not self.enabled or mode not in ("bench", "eager", "capture"):
            yield
            return
        key = f"{op_name}::{sig}"
        t0 = time.time()
        try:
            yield
        finally:
            dt_ms = (time.time() - t0) * 1000.0
            st = self.stats.setdefault(key, OpStat())
            st.calls += 1
            st.total_ms += dt_ms

    def maybe_report(self, every_sec: float = 2.0, topk: int = 10) -> None:
        if not self.enabled:
            return
        now = time.time()
        if now - self._last_report_t < every_sec:
            return
        self._last_report_t = now

        total = sum(st.total_ms for st in self.stats.values())
        if total <= 0:
            return

        items = sorted(self.stats.items(), key=lambda kv: kv[1].total_ms, reverse=True)[:topk]
        print("---- op breakdown ----")
        for k, st in items:
            share = (st.total_ms / total) * 100.0
            avg = st.total_ms / max(st.calls, 1)
            print(f"{share:6.2f}% | total_ms={st.total_ms:9.3f} | avg_ms={avg:7.3f} | calls={st.calls:6d} | {k}")
        print(f"TOTAL: {total:.3f} ms (accumulated)")
