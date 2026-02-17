from __future__ import annotations
from typing import Any, List, Optional, Dict

import torch
from .tensor_spec import TensorSpec
from .builder import Builder
from .layers.base import Layer
from .emitters.cuda.context import CudaEmitContext
from .runtime.cuda_exec import CudaExecutor  # 추가

class Model:
    def __init__(self, dtype: str = "f32", device: str = "cuda"):
        self.dtype = str(dtype)
        self.device = str(device)
        self.b = Builder(dtype=self.dtype, device=self.device)
        self.ctx = CudaEmitContext()
        
        # 순환 참조 방지를 위한 로컬 임포트
        from .runtime.cuda_exec import CudaExecutor
        self.executor = CudaExecutor() 
        self.compiled_program = None

    def input(self, name: str, spec: TensorSpec) -> int:
        """입력 텐서를 등록하고 Vid를 반환합니다."""
        spec = self._fill_spec_defaults(spec)
        return self.b.input(name, spec)

    def param(self, name: str, spec: TensorSpec) -> int:
        """학습 가능한 파라미터(Weight 등)를 등록합니다."""
        spec = self._fill_spec_defaults(spec)
        return self.b.param(name, spec)

    def state(self, name: str, spec: TensorSpec) -> int:
        """상태 값(Optimizer state, Step count 등)을 등록합니다."""
        spec = self._fill_spec_defaults(spec)
        return self.b.state(name, spec)

    def add(self, layer: Layer, *args, **kwargs) -> Any:
        """
        레이어의 emit 메서드를 호출하여 IR 그래프에 연산을 추가합니다.
        """
        return layer.emit(self.b, *args, ctx=self.ctx, **kwargs)

    def output(self, name: str, vid: int) -> None:
        """최종 출력 Vid를 지정합니다."""
        self.b.output(name, vid)

    def _fill_spec_defaults(self, spec: TensorSpec) -> TensorSpec:
        """TensorSpec에 누락된 dtype/device를 모델 기본값으로 채웁니다."""
        if spec.dtype is None or spec.device is None:
            return TensorSpec(
                shape=spec.shape,
                dtype=spec.dtype or self.dtype,
                device=spec.device or self.device,
            )
        return spec

    def dump(self) -> str:
        """현재 IR 그래프의 상태를 문자열로 출력합니다."""
        b = self.b
        in_names = [b.values[v].name for v in b.external_vids]
        out_names = [b.values[v].name for v in b.output_vids]

        lines = [f"[Graph] externals={in_names} outputs={out_names} ops={len(b.ops)}"]

        for i, op in enumerate(b.ops, start=1):
            ins = ", ".join(b.values[v].name for v in op.inputs)
            outs = ", ".join(b.values[v].name for v in op.outputs)

            extras = []
            if op.attrs: extras.append(f"attrs={op.attrs}")
            if op.constraints: extras.append(f"constraints={op.constraints}")
            if op.saved:
                saved_names = [b.values[v].name for v in op.saved]
                extras.append(f"saved={saved_names}")
            
            extra_str = ("  " + " ".join(extras)) if extras else ""
            lines.append(f"  o{i}: {op.kind}({ins}) -> {outs}{extra_str}")

        return "\n".join(lines)


class Sequential(Model):
    def __init__(self, layers: List[Layer], dtype: str = "f32", device: str = "cuda"):
        super().__init__(dtype=dtype, device=device)
        self.layers = layers
        self._is_built = False

    def build(self, input_spec: TensorSpec, input_name: str = "x") -> int:
        curr_vid = self.input(input_name, input_spec)
        for layer in self.layers:
            curr_vid = self.add(layer, curr_vid)
        self.output("output", curr_vid)
        self._is_built = True
        return curr_vid

    def compile(
        self, 
        registry: Optional[Any] = None, 
        capture: bool = False, 
        sample_feed: Optional[Dict[str, torch.Tensor]] = None,
        mode: str = "train"
    ):
        """
        [수정] capture=True 일 경우 컴파일 직후 CUDA Graph를 미리 생성합니다.
        """
        if not self._is_built:
            raise RuntimeError("Model must be built before compilation. Call build() first.")

        # 1. IR 수준 컴파일 (Plan 수립)
        self.compiled_program = self.executor.compile_cached(self)

        # 2. CUDA Graph 사전 캡처 (Pre-warmup)
        if capture:
            if sample_feed is None:
                raise ValueError("Graph capture를 위해선 sample_feed가 필요합니다.")
            
            # Executor를 통해 사전 캡처 수행
            # 이 과정에서 GPU 메모리 주소가 고정(Binding)됩니다.
            self.executor.capture_prebuilt(
                self, 
                self.compiled_program, 
                sample_feed, 
                mode=mode
            )
            print(f"[Model] CUDA Graph captured for mode='{mode}'")

        return self.compiled_program

    def run(self, feed: Dict[str, torch.Tensor], use_cuda_graph: bool = True, mode: str = "train"):
        """
        [신규] 캡처된 그래프를 사용하여 즉시 실행합니다.
        """
        if self.compiled_program is None:
            raise RuntimeError("Model must be compiled before running.")

        return self.executor.run_compiled(
            self,
            self.compiled_program,
            feed,
            use_cuda_graph=use_cuda_graph,
            mode=mode
        )