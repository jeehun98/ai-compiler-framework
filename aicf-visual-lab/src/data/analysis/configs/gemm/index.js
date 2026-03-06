// src/data/analysis/configs/gemm/index.js
import { gemmWmmaF16Config } from './wmma_f16.js';
import { gemmNaiveF32Config } from './naive_f32.js';

export const gemmAnalysis = {
  id: "gemm",
  label: "General Matrix Multiplication",
  category: "Linear Algebra",
  variants: [gemmWmmaF16Config, gemmNaiveF32Config],
  comparisonSummary:
  "WMMA 기반 FP16 GEMM은 naive FP32 GEMM보다 warp active와 SM throughput은 낮게 보일 수 있지만, Tensor Core 활용으로 실제 latency는 훨씬 짧습니다. 반대로 naive FP32는 장치를 더 빽빽하게 점유해 보이지만, scalar K-loop 구조 한계로 실행 시간은 더 길게 나타납니다."
};