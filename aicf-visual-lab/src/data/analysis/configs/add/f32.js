// src/data/analysis/configs/add/f32.js
import metrics from '../../metrics/add_f32.json';

export const f32Config = {
  id: "f32",
  name: "FP32 Baseline",
  tag: "Ref",
  description: "Standard single-precision floating point implementation.",
  metrics: metrics,
  features: ["FP32 Precision", "Full Range"],
  insights: [
    "High memory pressure due to 4-byte data width.",
    "Reference point for accuracy and baseline performance.",
    "Lowest throughput among tested variants."
  ]
};