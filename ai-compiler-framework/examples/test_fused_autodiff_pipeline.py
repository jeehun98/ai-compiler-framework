from __future__ import annotations

import numpy as np
from scipy import sparse
from typing import Any, List, Dict, Tuple
import sys
from pathlib import Path as _Path

# -----------------------
# project path bootstrap
# -----------------------
p = _Path(__file__).resolve()
root = next(parent for parent in [p] + list(p.parents) if (parent / "pyproject.toml").exists())
build_lib_path = root / "build" / "python" / "aicf_cuda"
src_path = root / "python" / "aicf_v2" / "src"
if build_lib_path.exists():
    sys.path.insert(0, str(build_lib_path))
sys.path.insert(0, str(src_path))

import aicf_v2 as aicf

# ---------------------------------------------------------
# 1. Fused Op Definition (BWD Responsibility)
# ---------------------------------------------------------
class FusedGemmBiasReluOp:
    KIND = "fused_gemm_bias_relu"
    KID = 100 

    @staticmethod
    def emit_bwd(b: Any, ctx: Any, fused_node: Any, grad_y: int) -> Dict[int, int]:
        """
        Fused 노드가 호출받았을 때 실행할 미분 로직.
        """
        print(f"\n >>> [BWD-INVOKE] Fused node '{fused_node.name}' is now generating its own BWD ops!")
        
        x_vid, w_vid, b_vid = fused_node.inputs[0], fused_node.inputs[1], fused_node.inputs[2]
        
        # 실제 미분 노드들을 Builder에 추가 (가상의 에미터 호출)
        grads = {
            x_vid: b.value(f"{fused_node.name}.grad_x", b.values[x_vid].spec),
            w_vid: b.value(f"{fused_node.name}.grad_w", b.values[w_vid].spec),
            b_vid: b.value(f"{fused_node.name}.grad_b", b.values[b_vid].spec),
        }
        
        # 여기서 추가적인 BWD Op을 emit할 수 있습니다.
        # 예: b.add_op(kind="fused_gemm_bwd", inputs=[grad_y, ...], ...)
        
        print(f" >>> [BWD-INVOKE] Gradients registered for 3 inputs.")
        return grads

# ---------------------------------------------------------
# 2. Advanced Rewriter (Injecting BWD Hook)
# ---------------------------------------------------------
class GraphOptimizerWithRewriter:
    def __init__(self, builder: Any):
        self.b = builder

    def apply_fusion_and_hook(self, patterns: List[Tuple[int, int, int]]):
        for i, j, k in patterns:
            op_g, op_b, op_r = self.b.ops[i], self.b.ops[j], self.b.ops[k]
            print(f"[REWRITE] Fusing: Op[{i}] -> Op[{j}] -> Op[{k}]")

            # FWD 통합
            op_g.kind = FusedGemmBiasReluOp.KIND
            op_g.kind_id = FusedGemmBiasReluOp.KID
            op_g.name = f"{op_g.name}_fused"
            op_g.inputs = [op_g.inputs[0], op_g.inputs[1], op_b.inputs[1]]
            op_g.outputs = op_r.outputs
            
            # [핵심] 런타임에 bwd_emit_fn 속성을 주입하여 model.py의 루프가 인식하게 함
            # 이 부분이 있어야 build_backward_from_ops 내부에서 커스텀 로직이 실행됩니다.
            setattr(op_g, 'bwd_emit_fn', FusedGemmBiasReluOp.emit_bwd)

            # 중간 노드 무효화
            for idx in [j, k]:
                self.b.ops[idx].kind = "nop"
                self.b.ops[idx].inputs, self.b.ops[idx].outputs = [], []

# ---------------------------------------------------------
# 3. Execution Pipeline
# ---------------------------------------------------------
def main():
    model = aicf.Sequential([
        aicf.Linear(784, 128, name="fc1"),
        aicf.ReLU(name="relu1"),
    ])
    
    x_spec = aicf.TensorSpec(shape=(64, 784), dtype="f32", device="cuda")
    y_vid = model.build(x_spec, input_name="x")
    
    # Scalar Loss 강제 생성 (Autodiff 시작점)
    from aicf_v2.emitters.cuda.reduce_sum import emit as emit_reduce
    final_loss_vid = model.b.value("final_loss", aicf.TensorSpec(shape=(1,), dtype="f32", device="cuda"))
    emit_reduce(model.b, model.ctx, x=y_vid, out=final_loss_vid, axis=0)

    # 행렬 최적화 및 리라이팅
    from test_rich_matrix_optimizer import RichMatrixOptimizer
    opt = RichMatrixOptimizer(model.b)
    opt.encode()
    targets = opt.find_fused_gemm_bias_relu()

    rewriter = GraphOptimizerWithRewriter(model.b)
    rewriter.apply_fusion_and_hook(targets)

    print("\n" + "="*80)
    print(" [PHASE] Starting Backward Generation")
    print("="*80)
    
    # Backward 생성 시 각 Op을 돌며 bwd_emit_fn이 있는지 확인합니다.
    # 만약 model.py가 이 필드를 지원하지 않는다면, 수동으로 루프를 돌려 호출할 수도 있습니다.
    model.build_backward_after_fwd_opt(final_loss_vid)

    print("\n[FINAL IR SNAPSHOT]")
    for i, op in enumerate(model.b.ops):
        status = " <--- FWD OPT" if op.kind == "fused_gemm_bias_relu" else ""
        print(f"Op[{i:02d}]: {op.kind:<25} {status} | Name: {op.name}")

if __name__ == "__main__":
    main()