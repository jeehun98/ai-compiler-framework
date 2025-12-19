# examples/python/aicf_fw/tensor.py
class Tensor:
    __slots__ = ("_ptr", "shape", "dtype", "device", "stride")

    # v0.1: ptr은 int(device address), device="cuda"만
    def __init__(self, ptr: int, shape: tuple[int, ...], dtype: str, stride=None, device="cuda"):
        self._ptr = ptr
        self.shape = tuple(shape)
        self.dtype = dtype  # "float16" | "float32"
        self.device = device
        self.stride = stride if stride is not None else self._contiguous_stride(self.shape)

    @property
    def data_ptr(self) -> int:
        return self._ptr

    @staticmethod
    def _contiguous_stride(shape):
        st = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            st[i] = st[i + 1] * shape[i + 1]
        return tuple(st)
