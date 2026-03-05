// src/data/analysis/configs/add/f16.js
import metrics from '../../metrics/add_f16.json';

export const f16Config = {
  id: "f16",
  name: "Naive FP16",
  tag: "Scalar",
  description: "Half precision scalar implementation without vectorization.",
  metrics: metrics,
  features: ["FP16 Precision", "Scalar Access"],
  insights: [
    "Baseline for half-precision performance.",
    "Instruction throughput is limited compared to vectorized paths.",
    "Memory bandwidth utilization is higher than FP32 but lower than f16x2."
  ]
};