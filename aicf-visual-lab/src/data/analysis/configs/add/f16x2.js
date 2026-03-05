// src/data/analysis/configs/add/f16x2.js
import metrics from '../../metrics/add_f16x2.json';

export const f16x2Config = {
  id: "f16x2",
  name: "Vectorized FP16 (half2)",
  tag: "Fast",
  description: "Using __half2 for vectorized 32-bit memory access and arithmetic.",
  metrics: metrics,
  features: ["Vectorized Load", "Register Re-use", "hadd2 Instruction"],
  insights: [
    "Reduces instruction issue overhead by 50%.",
    "Optimized for modern GPU L1/TEX cache paths.",
    "Significant throughput improvement for large element counts."
  ]
};