// src/data/analysis/configs/cross_entropy/f32.js
import metrics from '../../metrics/cross_entropy_f32.json';

export const crossEntropyF32Config = {
  id: "f32",
  name: "FP32 Cross Entropy",
  tag: "Reduction",
  description:
    "FP32 forward cross-entropy kernel using row-wise log-sum-exp reduction and scalar loss accumulation.",

  algorithm: {
    title: "Cross Entropy Forward (FP32 Log-Sum-Exp)",
    logic:
      "각 block이 하나의 sample row를 담당하며, logits에 대해 먼저 row max를 구한 뒤 exp(logit - max)의 합을 reduction으로 계산합니다. 이후 target class logit을 이용해 안정적인 cross-entropy loss를 만들고, 최종 loss 합과 valid count를 전역 accumulator에 반영합니다.",
    strategy: [
      "각 block이 sample row 하나를 처리",
      "row max 계산으로 수치 안정화",
      "exp-sum reduction으로 log-sum-exp 구성",
      "target class 기준 scalar loss 계산",
      "loss_sum, valid에 atomicAdd 후 finalize kernel에서 mean/sum 처리"
    ]
  },

  blueprint: {
    mem_access: "32-bit row-wise streaming load + scalar atomic accumulation",
    instruction: "expf + logf + warp/block reduction + atomicAdd",
    vector_width: 1,
    alignment_req: "4-byte",
    code_snippet:
      "m = block_max(m); s = block_sum(expf(row[c] - m)); if (threadIdx.x == 0) { atomicAdd(out_loss, logf(s) + m - row[t]); atomicAdd(out_valid, 1); }"
  },

  metrics,
  features: [
    "FP32 Logits",
    "Log-Sum-Exp Stabilization",
    "Warp/Block Reduction",
    "Scalar Loss Accumulation",
    "Forward Loss Kernel"
  ],
  insights: [
    "이 측정의 핵심 대상은 cross-entropy forward의 row-wise reduction 커널입니다.",
    "BatchNorm stats처럼 원소 단위 atomic contention이 지배적인 구조는 아니며, reduction과 exp/log 연산 비용이 더 큰 비중을 가집니다.",
    "warp active는 높게 유지되지만, pure pointwise 커널 대비 throughput은 reduction 및 transcendental 연산 영향으로 제한됩니다."
  ]
};