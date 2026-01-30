from __future__ import annotations

class Layer:
    def __init__(self, name: str):
        self.name = str(name)

    def emit(self, b, *inputs: int) -> int:
        raise NotImplementedError
