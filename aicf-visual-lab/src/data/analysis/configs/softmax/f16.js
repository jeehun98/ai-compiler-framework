// src/data/analysis/configs/softmax/f16.js
import metrics from '../../metrics/softmax_f16.json';

export const softmaxF16Config = {
  id: "f16",
  name: "FP16 Softmax",
  tag: "Fast",
  description:
    "Half-precision I/O softmax over the last dimension, using FP32 reduction and exponentiation internally for stability and efficiency.",

  algorithm: {
    title: "Last-Dimension Softmax (FP16 I/O, FP32 Compute)",
    logic:
      "각 block이 하나의 row를 담당하며, 마지막 차원에 대해 max reduction, exp-sum reduction, normalization을 순차적으로 수행합니다. 입력과 출력은 FP16이지만 내부의 max/sum/exp 계산은 float로 수행되어 수치 안정성과 구현 단순성을 유지합니다.",
    strategy: [
      "row-wise last-dimension softmax",
      "입력/출력은 FP16, 내부 reduction/exp는 FP32",
      "warp shuffle 기반 reduction + shared scratch 결합",
      "한 row에 대해 max / sum / write의 multi-pass 구조"
    ]
  },

  blueprint: {
    mem_access: "16-bit row-wise streaming load/store",
    instruction: "__half2float + __expf + warp/block reduction",
    vector_width: 1,
    alignment_req: "2-byte",
    code_snippet:
      "float v = __half2float(x[base + c]); tsum += __expf(v - rmax); y[base + c] = __float2half_rn(__expf(v - rmax) * inv);"
  },

  metrics,
  features: [
    "FP16 Input/Output",
    "FP32 Reduction",
    "Warp Shuffle Reduction",
    "Row-wise Softmax"
  ],
  insights: [
    "FP16 I/O를 사용하지만 내부의 max, sum, exp 계산은 float로 수행됩니다.",
    "FP32 대비 warp active는 거의 동일하지만 더 높은 SM throughput과 더 짧은 latency를 보였습니다.",
    "데이터 폭 감소로 global memory traffic이 줄어드는 점이 성능 향상에 기여했을 가능성이 큽니다."
  ]
};