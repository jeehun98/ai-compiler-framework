// src/data/analysis/configs/softmax/f32.js
import metrics from '../../metrics/softmax_f32.json';

export const softmaxF32Config = {
  id: "f32",
  name: "FP32 Softmax",
  tag: "Baseline",
  description:
    "Single-precision baseline softmax over the last dimension using FP32 reduction, exponentiation, and output.",

  algorithm: {
    title: "Last-Dimension Softmax (FP32)",
    logic:
      "각 block이 하나의 row를 담당하며, 마지막 차원 기준으로 max reduction, exp-sum reduction, normalization을 수행합니다. warp shuffle과 block reduction을 조합한 전형적인 row-wise softmax 구현입니다.",
    strategy: [
      "row-wise last-dimension softmax",
      "입력/출력/내부 계산 모두 FP32",
      "warp shuffle 기반 reduction + shared scratch 사용",
      "한 row에 대해 max / sum / write의 multi-pass 구조"
    ]
  },

  blueprint: {
    mem_access: "32-bit row-wise streaming load/store",
    instruction: "__expf + warp/block reduction",
    vector_width: 1,
    alignment_req: "4-byte",
    code_snippet:
      "tsum += __expf(x[base + c] - rmax); y[base + c] = __expf(x[base + c] - rmax) * inv;"
  },

  metrics,
  features: [
    "FP32 Precision",
    "FP32 Reduction",
    "Warp Shuffle Reduction",
    "Baseline Reference"
  ],
  insights: [
    "FP16과 거의 동일한 block 구조와 reduction 패턴을 사용합니다.",
    "더 큰 데이터 폭으로 인해 memory traffic 부담이 커지고 latency가 더 길게 나타났습니다.",
    "수치적 기준선 역할을 하는 baseline softmax 구현입니다."
  ]
};