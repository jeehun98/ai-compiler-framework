// src/data/analysis/configs/softmax/f16.js
import metrics from '../../metrics/softmax_f16.json';

export const softmaxF16Config = {
  id: "f16",
  name: "FP16 Softmax",
  tag: "Fast",
  description:
    "Half-precision softmax over the last dimension with reduced memory footprint and improved execution efficiency.",

  algorithm: {
    title: "Last-Dimension FP16 Softmax",
    logic:
      "각 block이 마지막 차원에 대한 softmax를 계산하며, FP16 입력/출력을 사용해 데이터 이동량을 줄입니다. reduction, exp, normalization 단계는 FP32와 동일하지만 더 작은 데이터 폭으로 인해 처리 효율이 향상됩니다.",
    strategy: [
      "마지막 차원 기준 row-wise softmax 수행",
      "FP16 입력 / FP16 출력",
      "reduction + exp + normalization 구조",
      "낮은 메모리 footprint로 bandwidth 부담 감소"
    ]
  },

  blueprint: {
    mem_access: "16-bit scalar access",
    instruction: "FP16 arithmetic + reduction",
    vector_width: 1,
    alignment_req: "2-byte",
    code_snippet:
      "out[idx] = exp(x[idx] - row_max) / row_sum; // FP16 softmax"
  },

  metrics,
  features: [
    "FP16 Precision",
    "Reduced Memory Footprint",
    "Row-wise Reduction",
    "Higher Throughput"
  ],
  insights: [
    "FP32 대비 warp active는 거의 동일하지만 더 높은 SM throughput을 보입니다.",
    "데이터 폭 감소로 인해 메모리 및 연산 비용이 낮아져 latency가 단축됩니다.",
    "실측 결과 기준 FP32 대비 약 1.34배 더 짧은 실행 시간을 보였습니다."
  ]
};