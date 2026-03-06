// src/data/analysis/configs/gemm/f32.js
import metrics from '../../metrics/gemm_f32.json';

export const gemmNaiveF32Config = {
  id: "f32",
  name: "FP32 Naive GEMM",
  tag: "SIMT",
  description:
    "Naive FP32 GEMM where each thread computes one output element using a scalar K-loop on standard CUDA cores.",

  algorithm: {
    title: "Naive SIMT GEMM",
    logic:
      "각 thread가 C[row, col]의 단일 원소를 담당하고, K 전체를 순회하며 A[row, k] * B[k, col]를 누산합니다. shared memory 타일링 없이 global memory를 직접 참조하는 가장 단순한 GEMM 형태입니다.",
    strategy: [
      "각 thread가 출력 원소 하나 계산",
      "K loop를 scalar FMA 형태로 순차 누산",
      "Shared memory 미사용",
      "표준 CUDA core 기반 FP32 연산"
    ]
  },

  blueprint: {
    mem_access: "32-bit scalar global load",
    instruction: "fma.rn.f32 / scalar multiply-add",
    vector_width: 1,
    alignment_req: "4-byte",
    code_snippet:
      "acc += A[row * Ars + k * Acs] * B[k * Brs + col * Bcs];"
  },

  metrics: metrics,
  features: [
    "FP32 Precision",
    "Naive K-loop",
    "Per-thread Output Element",
    "SIMT Execution"
  ],
  insights: [
    "warp active와 SM throughput은 매우 높게 나타날 수 있지만, 실행 시간 자체는 WMMA 기반 FP16보다 불리할 수 있습니다.",
    "각 thread가 global memory를 반복 참조하므로 데이터 재사용 측면에서 비효율적입니다.",
    "baseline 구현으로는 적절하지만 고성능 GEMM 구조로 보기는 어렵습니다."
  ]
};