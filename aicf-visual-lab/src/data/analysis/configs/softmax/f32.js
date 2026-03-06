// src/data/analysis/configs/softmax/f32.js
import metrics from '../../metrics/softmax_f32.json';

export const softmaxF32Config = {
  id: "f32",
  name: "FP32 Softmax",
  tag: "Baseline",
  description:
    "Single-precision baseline softmax over the last dimension with higher numerical robustness and larger data footprint.",

  algorithm: {
    title: "Last-Dimension FP32 Softmax",
    logic:
      "각 block이 마지막 차원 기준 softmax를 수행하며, reduction과 normalization 구조는 FP16과 동일하지만 FP32 데이터 폭으로 인해 더 큰 메모리 트래픽과 연산 비용이 발생합니다.",
    strategy: [
      "마지막 차원 기준 row-wise softmax 수행",
      "FP32 입력 / FP32 출력",
      "reduction + exp + normalization 구조",
      "수치 안정성 기준 baseline"
    ]
  },

  blueprint: {
    mem_access: "32-bit scalar access",
    instruction: "FP32 arithmetic + reduction",
    vector_width: 1,
    alignment_req: "4-byte",
    code_snippet:
      "out[idx] = expf(x[idx] - row_max) / row_sum; // FP32 softmax"
  },

  metrics,
  features: [
    "FP32 Precision",
    "Baseline Reference",
    "Row-wise Reduction",
    "Numerical Stability"
  ],
  insights: [
    "FP16과 유사한 warp active를 보이지만 SM throughput은 더 낮습니다.",
    "더 큰 데이터 폭으로 인해 execution latency가 FP16 대비 길게 나타났습니다.",
    "수치 안정성 관점의 기준점으로 적합한 구현입니다."
  ]
};