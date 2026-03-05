// data/analysis/index.js
import add_f32 from './metrics/add_f32.json';
import add_f16 from './metrics/add_f16.json';
import add_f16x2 from './metrics/add_f16x2.json';

export const analysisConfigs = {
  add: {
    title: "Element-wise Add",
    summary: "FP16 Vectorization(half2)을 통한 메모리 대역폭 효율 최적화 실험",
    variants: [
      { id: "f32", label: "FP32 Baseline", data: add_f32 },
      { id: "f16", label: "Naive FP16", data: add_f16 },
      { id: "f16x2", label: "Vectorized FP16", data: add_f16x2 }
    ]
  }
};