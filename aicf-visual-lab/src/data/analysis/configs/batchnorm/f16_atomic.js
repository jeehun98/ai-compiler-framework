// src/data/analysis/configs/batchnorm/f16_atomic.js
import metrics from '../../metrics/batchnorm_f16_atomic.json';

export const batchNormF16AtomicConfig = {
  id: "f16_atomic",
  name: "FP16 BatchNorm Stats (Atomic)",
  tag: "Hotspot",
  description:
    "Hotspot-focused analysis of the BatchNorm forward statistics stage. This kernel scans FP16 input and accumulates per-channel sum and sumsq in FP32 using atomicAdd, making it a strong bottleneck candidate in the current training pipeline.",

  algorithm: {
    title: "BatchNorm Stats Extraction (FP16 Input, FP32 Atomic Accumulation)",
    logic:
      "현재 분석 대상은 BatchNorm forward 전체가 아니라 stats 추출 단계입니다. 입력 x를 N*C*HW 전체 범위로 순회하면서 각 원소를 channel c에 매핑하고, sum[c]와 sumsq[c]에 FP32 atomicAdd를 수행합니다. 이후 mean/variance finalize 및 normalization apply 단계의 입력이 되는 통계를 생성합니다.",
    strategy: [
      "BatchNorm 전체 파이프라인 중 stats 커널만 분리 분석",
      "N*C*HW 전체 원소를 grid-stride loop로 순회",
      "각 원소를 channel index c로 매핑",
      "sum[c], sumsq[c]를 FP32 atomicAdd로 누적",
      "병목 후보를 먼저 정밀 분석하는 hotspot-first 접근"
    ]
  },

  blueprint: {
    mem_access: "16-bit streaming load + FP32 atomic accumulation",
    instruction: "__half2float + atomicAdd(float)",
    vector_width: 1,
    alignment_req: "2-byte",
    code_snippet:
      "const int c = (int)((i / HW) % C); const float v = __half2float(x[i]); atomicAdd(&sum[c], v); atomicAdd(&sumsq[c], v * v);"
  },

  metrics,
  features: [
    "FP16 Input",
    "FP32 Accumulation",
    "Atomic Reduction",
    "Channel-wise Statistics",
    "Hotspot Analysis"
  ],
  insights: [
    "이 측정은 BatchNorm 전체 forward가 아니라 stats accumulation 커널 하나에 대한 결과입니다.",
    "현재 구현에서는 이 stats 단계가 가장 유력한 병목 후보이며, 전체 파이프라인보다 먼저 정밀 분석할 가치가 큽니다.",
    "매 원소마다 atomicAdd를 2회 수행하므로 channel-wise write contention이 발생할 수 있습니다.",
    "낮은 SM throughput은 단순 연산 부족보다 atomic accumulation의 serialization 비용을 시사합니다."
  ]
};