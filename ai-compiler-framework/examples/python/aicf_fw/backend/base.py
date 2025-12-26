# aicf_fw/backend/base.py
from __future__ import annotations
from typing import Any, Dict, List

class Backend:
    def op_call(self, op: str, inputs: List[Any], attrs: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def ones_like(self, x: Any) -> Any:
        raise NotImplementedError
