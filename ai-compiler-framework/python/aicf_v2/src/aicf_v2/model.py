from __future__ import annotations
from typing import Any, List, Optional, Dict, TYPE_CHECKING

import torch
from .tensor_spec import TensorSpec
from .builder import Builder
from .layers.base import Layer
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

        # 자동 미분 및 그래프 관리
        self._tape: List[Dict[str, Any]] = [] 
        self.parameter_grads: Dict[int, int] = {} 
        
        # 실제 데이터 저장소 (Key: 변수명, Value: torch.Tensor)
        self.parameters: Dict[str, torch.Tensor] = {}
        self.states: Dict[str, torch.Tensor] = {}

    def _fill_spec_defaults(self, spec: TensorSpec) -> TensorSpec:
        """TensorSpec의 None인 필드를 모델의 기본값(cuda, f32)으로 채웁니다."""
        return TensorSpec(
            shape=spec.shape,
            dtype=spec.dtype or self.dtype,
            device=spec.device or self.device,
        )

    def input(self, name: str, spec: TensorSpec) -> int:
        return self.b.input(name, self._fill_spec_defaults(spec))

    def param(self, name: str, spec: TensorSpec) -> int:
        """파라미터를 등록하고 실제 텐서를 할당하여 저장소에 보관합니다."""
        spec = self._fill_spec_defaults(spec)
        vid = self.b.param(name, spec)
        
        # [중요] 레이어 내부에서 param을 생성할 때 이 메서드를 거치도록 유도하거나 
        # 호출 시점에 텐서를 즉시 생성합니다.
        if name not in self.parameters:
            t = torch.randn(spec.shape, dtype=torch.float32, device=self.device)
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
        """레이어를 IR에 추가하고 파라미터 변화를 감지하여 테이프에 기록합니다."""
        prev_param_vids = set(self.b.param_vids)
        
        # 레이어 emit 시 내부에서 b.param을 호출하면 
        # 우리는 Model.param을 거치도록 설계하거나 사후에 등록해야 합니다.
        out_vids = layer.emit(self.b, *args, ctx=self.ctx, **kwargs)
        
        # [신규] 레이어 emit 도중 Builder에 새로 추가된 파라미터들을 
        # 모델의 실제 텐서 저장소(self.parameters)에 자동으로 바인딩합니다.
        for vid in self.b.param_vids:
            val = self.b.values[vid]
            if val.name not in self.parameters:
                # Model.param 로직을 재활용하여 텐서 생성
                self.param(val.name, val.spec)

        new_params = [vid for vid in self.b.param_vids if vid not in prev_param_vids]

        self._tape.append({
            "type": "layer", 
            "layer": layer, 
            "inputs": args,
            "outputs": [out_vids] if isinstance(out_vids, int) else list(out_vids),
            "params": new_params
        })
        return out_vids

    def op(self, kind: str, inputs: List[int], outputs: Any, name: Optional[str] = None) -> int:
        filled_outputs = []
        if isinstance(outputs, list):
            for o in outputs:
                filled_outputs.append(self._fill_spec_defaults(o) if isinstance(o, TensorSpec) else o)
        else:
            filled_outputs = self._fill_spec_defaults(outputs) if isinstance(outputs, TensorSpec) else outputs

        out_vid = self.b.op(kind, inputs, filled_outputs, name)
        self._tape.append({
            "type": "op", "kind": kind, "inputs": inputs,
            "outputs": [out_vid] if isinstance(out_vid, int) else list(out_vid)
        })
        return out_vid

    def build_backward(self, loss_vid: int) -> Dict[int, int]:
        loss_spec = self.b.values[loss_vid].spec
        initial_grad_spec = self._fill_spec_defaults(loss_spec)
        grad_map = {loss_vid: self.b.input("grad_initial", initial_grad_spec)}

        for entry in reversed(self._tape):
            outs = entry["outputs"]
            if not outs or outs[0] not in grad_map: 
                continue
                
            grad_y = grad_map[outs[0]]
            ins = entry["inputs"]

            if entry["type"] == "layer":
                layer = entry["layer"]
                params = entry["params"]
                
                if hasattr(layer, "emit_backward"):
                    layer_name = layer.__class__.__name__
                    
                    if layer_name == "Linear":
                        bwd_args = {
                            "b": self.b, "x": ins[0], "W": params[0], 
                            "grad_y": grad_y, "ctx": self.ctx
                        }
                        if len(params) > 1: bwd_args["bias"] = params[1]
                    elif layer_name == "MSELoss":
                        bwd_args = {
                            "b": self.b, "y_pred": ins[0], "y_true": ins[1],
                            "grad_y": grad_y, "ctx": self.ctx
                        }
                    else:
                        bwd_args = {"b": self.b, "ctx": self.ctx, "grad_y": grad_y}

                    layer_grads = layer.emit_backward(**bwd_args)
                    
                    if "input" in layer_grads:
                        grad_map[ins[0]] = layer_grads["input"]
                    
                    if "weight" in layer_grads and len(params) > 0:
                        self.parameter_grads[params[0]] = layer_grads["weight"]
                    if "bias" in layer_grads and len(params) > 1:
                        self.parameter_grads[params[1]] = layer_grads["bias"]
            
            elif entry["type"] == "op":
                if entry["kind"] in ["sum", "sub", "add"]:
                    grad_map[ins[0]] = grad_y

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
            curr_vid = self.add(layer, curr_vid)
        self.output("output", curr_vid)
        self._is_built = True
        return curr_vid

    def compile(self, registry: Optional[Any] = None, capture: bool = False, 
                sample_feed: Optional[Dict[str, torch.Tensor]] = None, mode: str = "train"):
        if not self._is_built:
            raise RuntimeError("Model must be built before compilation.")

        # 1. 통합 피드 생성 (이때 fc1.W 등이 확실히 포함됩니다)
        full_sample_feed = self.get_full_feed(sample_feed or {})

        self.compiled_program = self.executor.compile_cached(self)

        if capture:
            # 2. 캡처 수행
            self.executor.capture_prebuilt(self, self.compiled_program, full_sample_feed, mode=mode)
            print(f"[Model] CUDA Graph captured for mode='{mode}'")

        return self.compiled_program
    
    def run(self, feed: Dict[str, torch.Tensor], use_cuda_graph: bool = True, mode: str = "train"):
        if self.compiled_program is None: 
            raise RuntimeError("Model must be compiled before running.")
        return self.executor.run_compiled(self, self.compiled_program, feed, use_cuda_graph=use_cuda_graph, mode=mode)