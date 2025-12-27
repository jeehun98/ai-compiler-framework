# aicf_fw/backend/base.py
from __future__ import annotations
from typing import Any, Dict, List

class Backend:
    def op_call(self, op: str, inputs: List[Any], attrs: Dict[str, Any]) -> Any:
        raise NotImplementedError

    # ✅ out-parameter 버전 (in-place or explicit outputs)
    def op_call_out(self, op: str, inputs: List[Any], outputs: List[Any], attrs: Dict[str, Any]) -> None:
        raise NotImplementedError

    def ones_like(self, x: Any) -> Any:
        raise NotImplementedError

    def capture_begin(self):
        pass

    def capture_end(self):
        pass

    def replay(self):
        pass

    def capture_reset(self):
        pass
