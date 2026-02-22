from __future__ import annotations
from typing import Any, List, Optional, Dict, TYPE_CHECKING, Iterable

import torch

from .tensor_spec import TensorSpec
from .builder import Builder
from .layers.base import Layer

# 런타임에 CudaEmitContext 객체를 생성해야 하므로 TYPE_CHECKING 밖으로 유지
from .emitters.cuda.context import CudaEmitContext

if TYPE_CHECKING:
    from .runtime.cuda_exec import CudaExecutor

# grad accumulation에서 매번 import 하지 않도록 모듈 레벨로 올림
from .emitters.cuda import add as emit_add


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
            if "bias" in name or name.endswith(".b") or ".bias" in name:
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
        """
        레이어의 emit을 실행하고, "이번 emit에서 새로 추가된 파라미터만" 자동 바인딩합니다.
        (기존: 매번 전체 param_vids를 순회 -> 모델 커지면 O(N^2)로 악화 가능)
        """
        before = len(self.b.param_vids)
        out_vids = layer.emit(self.b, *args, ctx=self.ctx, **kwargs)

        # Builder에 새로 추가된 파라미터만 감지 및 할당
        new_param_vids = self.b.param_vids[before:]
        for vid in new_param_vids:
            val = self.b.values[vid]
            if val.name not in self.parameters:
                self.param(val.name, val.spec)

        return out_vids

    # -----------------------------
    # Backward builders
    # -----------------------------
    def _init_grad_map_for_loss(self, loss_vid: int) -> Dict[int, int]:
        """loss_vid로부터 grad_initial을 만들고 grad_map을 초기화합니다."""
        loss_val = self.b.values[loss_vid]
        loss_spec = self._fill_spec_defaults(loss_val.spec)

        # loss는 scalar-like 가정
        if loss_spec.shape not in ((), (1,)):
            raise RuntimeError(
                f"Loss must be scalar-like for current backward builder, got shape={loss_spec.shape}. "
                f"Make sure your loss reduction produces a scalar."
            )

        grad_initial_vid = self.b.input("grad_initial", loss_spec)
        return {loss_vid: grad_initial_vid}

    def build_backward_from_ops(self, fwd_ops: Iterable[Any], loss_vid: int) -> Dict[int, int]:
        """
        (구조 완성용)
        '특정 op 리스트'를 기준으로 역전파 그래프를 생성합니다.
        - 최적화(정규화) 이후의 fwd op snapshot을 넣으면,
          fused op는 fused emit_bwd로 처리되는 구조가 됩니다.
        """
        grad_map = self._init_grad_map_for_loss(loss_vid)

        for node in reversed(list(fwd_ops)):
            active_outputs = [out_vid for out_vid in node.outputs if out_vid in grad_map]
            if not active_outputs:
                continue

            grad_y = grad_map[active_outputs[0]]
            node_grads = self.ctx.emit_bwd_for_node(self.b, node, grad_y)

            for in_vid, g_vid in node_grads.items():
                if in_vid in grad_map:
                    existing_g = grad_map[in_vid]
                    combined_name = f"grad_acc_{in_vid}"
                    acc_vid = self.b.value(combined_name, self.b.values[in_vid].spec)

                    emit_add.emit(
                        self.b,
                        self.ctx,
                        a=existing_g,
                        b=g_vid,
                        out=acc_vid,
                        name=combined_name,
                    )
                    grad_map[in_vid] = acc_vid
                else:
                    grad_map[in_vid] = g_vid

                if in_vid in self.b.param_vids:
                    self.parameter_grads[in_vid] = grad_map[in_vid]

        return grad_map

    def build_backward(self, loss_vid: int) -> Dict[int, int]:
        """
        (기존 동작 유지)
        Builder의 ops 전체를 역순회하며 미분 그래프를 생성합니다.
        """
        return self.build_backward_from_ops(self.b.ops, loss_vid)
    
    def build_backward_after_fwd_opt(self, loss_vid: int) -> Dict[int, int]:
        """
        (논리 구조)
        1) fwd를 먼저 optimize 단계(현재 identity)로 통과
        2) optimize된 fwd ops snapshot을 잡고
        3) 그 snapshot을 기준으로 bwd를 생성
        """
        # optimize hook (현재 identity여도 OK)
        from .compile.passes.pipeline import optimize_ir
        optimize_ir(self.b)

        fwd_ops_snapshot = list(self.b.ops)
        return self.build_backward_from_ops(fwd_ops_snapshot, loss_vid)

    # -----------------------------
    # feeds / outputs
    # -----------------------------
    def get_full_feed(self, user_feed: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        full_feed: Dict[str, torch.Tensor] = {}
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

    def compile(
        self,
        capture: bool = False,
        sample_feed: Optional[Dict[str, torch.Tensor]] = None,
        mode: str = "train",
    ):
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

        full_feed = self.get_full_feed(feed)
        return self.executor.run_compiled(
            self,
            self.compiled_program,
            full_feed,
            use_cuda_graph=use_cuda_graph,
            mode=mode,
        )