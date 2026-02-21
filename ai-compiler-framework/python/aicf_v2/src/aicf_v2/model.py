from __future__ import annotations
from typing import Any, List, Optional, Dict, TYPE_CHECKING

import torch
from .tensor_spec import TensorSpec
from .builder import Builder
from .layers.base import Layer

# 런타임에 CudaEmitContext 객체를 생성해야 하므로 TYPE_CHECKING 밖으로 유지
from .emitters.cuda.context import CudaEmitContext 

if TYPE_CHECKING:
    from .runtime.cuda_exec import CudaExecutor
    
class Model:
    def __init__(self, dtype: str = "f32", device: str = "cuda"):
        self.dtype = str(dtype)
        self.device = str(device)
        self.b = Builder(dtype=self.dtype, device=self.device)
        self.ctx = CudaEmitContext()
        
        from .runtime.cuda_exec import CudaExecutor
        self.executor = CudaExecutor() 
        self.compiled_program = None

        # 파라미터 미분값 Vid 매핑 (Optimizer용)
        # Key: Parameter Vid, Value: Gradient Vid
        self.parameter_grads: Dict[int, int] = {} 
        
        # 실제 데이터 저장소
        self.parameters: Dict[str, torch.Tensor] = {}
        self.states: Dict[str, torch.Tensor] = {}

    def _fill_spec_defaults(self, spec: TensorSpec) -> TensorSpec:
        return TensorSpec(
            shape=spec.shape,
            dtype=spec.dtype or self.dtype,
            device=spec.device or self.device,
        )

    def input(self, name: str, spec: TensorSpec) -> int:
        return self.b.input(name, self._fill_spec_defaults(spec))

    def param(self, name: str, spec: TensorSpec) -> int:
        spec = self._fill_spec_defaults(spec)
        vid = self.b.param(name, spec)
        if name not in self.parameters:
            # Xavier/Kaiming 초기화 대신 단순 랜덤 (편의상)
            t = torch.randn(spec.shape, dtype=torch.float32, device=self.device) * 0.01
            if "bias" in name or ".b" in name:
                t.zero_()
            self.parameters[name] = t
        return vid

    def state(self, name: str, spec: TensorSpec) -> int:
        spec = self._fill_spec_defaults(spec)
        vid = self.b.state(name, spec)
        if name not in self.states:
            self.states[name] = torch.zeros(spec.shape, dtype=torch.float32, device=self.device)
        return vid

    def add(self, layer: Layer, *args, **kwargs) -> Any:
        """레이어의 emit을 실행하고 파라미터를 자동 바인딩합니다."""
        out_vids = layer.emit(self.b, *args, ctx=self.ctx, **kwargs)
        
        # Builder에 새로 추가된 파라미터 감지 및 할당
        for vid in self.b.param_vids:
            val = self.b.values[vid]
            if val.name not in self.parameters:
                self.param(val.name, val.spec)
        return out_vids

    def build_backward(self, loss_vid: int) -> Dict[int, int]:
        """
        [혁신] Builder의 Op 기록(ops)을 역순회하며 미분 그래프를 자동 누적합니다.
        기존 emit_nodes 에러를 해결하기 위해 b.ops를 사용합니다.
        """
        # 1. 초기 미분값(Loss에 대한 미분) 설정
        loss_val = self.b.values[loss_vid]
        grad_initial_spec = self._fill_spec_defaults(loss_val.spec)
        # 통상적으로 loss는 scalar이므로 grad_initial은 1.0으로 채워진 input이 됨
        grad_map = {loss_vid: self.b.input("grad_initial", grad_initial_spec)}

        # 2. Builder에 기록된 Op(Emit Node)들을 역순으로 순회 (Mirroring)
        # [수정] self.b.emit_nodes 대신 self.b.ops를 순회합니다.
        for node in reversed(self.b.ops):
            # 이 노드의 출력들 중 하나라도 미분 대상(grad_map에 존재)인 경우 역연산 수행
            # 단일 출력 노드뿐 아니라 BatchNorm 등 다중 출력 노드도 대응
            active_outputs = [out_vid for out_vid in node.outputs if out_vid in grad_map]
            
            if not active_outputs:
                continue

            # 주 미분값(dy) 추출 (일반적으로 첫 번째 출력 미분값 사용)
            grad_y = grad_map[active_outputs[0]]
            
            # [Dynamic Dispatch] Context가 node.kind를 보고 적절한 Emitter 모듈의 emit_bwd를 실행
            # 예: node.kind="gemm" -> gemm.emit_bwd 호출
            node_grads = self.ctx.emit_bwd_for_node(self.b, node, grad_y)

            # 3. 계산된 입력들에 대한 미분값을 grad_map에 갱신 (전파)
            for in_vid, g_vid in node_grads.items():
                if in_vid in grad_map:
                    # [Lattice Optimization] 이미 미분값이 있다면 EltwiseAdd 노드를 추가하여 합산
                    # 이는 텐서가 여러 노드에 입력으로 쓰였을 때(Branch) 필수적인 Gradient Accumulation 로직입니다.
                    existing_g = grad_map[in_vid]
                    combined_name = f"grad_acc_{in_vid}"
                    
                    # 수동으로 EltwiseAdd를 빌드하거나, 통합된 add 이미터를 사용할 수 있습니다.
                    from .emitters.cuda import add as emit_add
                    acc_vid = self.b.value(combined_name, self.b.values[in_vid].spec)
                    emit_add.emit(self.b, self.ctx, a=existing_g, c=g_vid, out=acc_vid, name=combined_name)
                    grad_map[in_vid] = acc_vid
                else:
                    grad_map[in_vid] = g_vid
                
                # 4. 파라미터(Weight/Bias)에 대한 미분인 경우 Optimizer가 찾을 수 있게 별도 보관
                if in_vid in self.b.param_vids:
                    self.parameter_grads[in_vid] = grad_map[in_vid]

        return grad_map

    def get_full_feed(self, user_feed: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        full_feed = {}
        full_feed.update(self.parameters)
        full_feed.update(self.states)
        full_feed.update(user_feed)
        return full_feed

    def output(self, name: str, vid: int) -> None:
        self.b.output(name, vid)

class Sequential(Model):
    def __init__(self, layers: List[Layer], dtype: str = "f32", device: str = "cuda"):
        super().__init__(dtype=dtype, device=device)
        self.layers = layers
        self._is_built = False

    def build(self, input_spec: TensorSpec, input_name: str = "x") -> int:
        curr_vid = self.input(input_name, input_spec)
        for layer in self.layers:
            # add 메서드 내부에서 layer.emit과 파라미터 바인딩이 일어남
            curr_vid = self.add(layer, curr_vid)
        
        self.output("output", curr_vid)
        self._is_built = True
        return curr_vid

    def compile(self, capture: bool = False, 
                sample_feed: Optional[Dict[str, torch.Tensor]] = None, mode: str = "train"):
        if not self._is_built:
            raise RuntimeError("Model must be built before compilation.")

        full_sample_feed = self.get_full_feed(sample_feed or {})
        
        # Executor가 Builder의 ops를 바탕으로 실제 실행 가능한 커널 프로그램을 생성
        self.compiled_program = self.executor.compile_cached(self)

        if capture:
            self.executor.capture_prebuilt(self, self.compiled_program, full_sample_feed, mode=mode)
            print(f"[Model] CUDA Graph captured for mode='{mode}'")

        return self.compiled_program
    
    def run(self, feed: Dict[str, torch.Tensor], use_cuda_graph: bool = True, mode: str = "train"):
        if self.compiled_program is None: 
            raise RuntimeError("Model must be compiled before running.")
        
        # 실시간 입력(user_feed)과 내부 상태(param, state)를 병합하여 실행
        full_feed = self.get_full_feed(feed)
        return self.executor.run_compiled(self, self.compiled_program, full_feed, use_cuda_graph=use_cuda_graph, mode=mode)