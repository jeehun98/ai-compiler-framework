export const biasAddDeepDive = {
  id: "BiasAdd",
  kernel_evolution: [
    {
      version: "v1.0",
      tag: "Scalar Load/Store",
      throughput: "—",
      description: "각 element에 대해 bias를 더하는 기본 구현. 메모리 접근이 주 병목.",
    },
    {
      version: "v2.0",
      tag: "Half2 Vectorization",
      throughput: "—",
      description: "N이 even일 때 half2로 벡터화하여 load/add/store 처리. 처리량 개선.",
    },
  ],
  profiling_report: {
    "DRAM_Throughput": "—",
    "Warp_Execution_Efficiency": "—",
    "L1_Cache_Hit_Rate": "—",
  },
  analysis:
    "축(last-dim) broadcast만 지원. fp16에서는 N even이면 half2 경로로 벡터화, odd이면 scalar half 경로로 폴백.",
  tests: {
    summary: "CUDA binding test: exact match vs torch reference (max|delta| = 0).",
    positive: [
      { dtype: "fp32", shape: [64, 256], axis: -1, max_abs_delta: 0.0 },
      { dtype: "fp32", shape: [8, 32, 128], axis: -1, max_abs_delta: 0.0 },
      { dtype: "fp16", shape: [64, 256], axis: -1, max_abs_delta: 0.0, note: "N even => half2 가능" },
      { dtype: "fp16", shape: [7, 33, 127], axis: -1, max_abs_delta: 0.0, note: "N odd => scalar half 폴백" },
    ],
    negative: [
      { name: "axis not last-dim", axis: 1, expected_status: "NotImplemented" },
      { name: "bias length mismatch", expected_status: "InvalidArgument (권장)", got_status: "NotImplemented (현재)" },
    ],
  },
};
