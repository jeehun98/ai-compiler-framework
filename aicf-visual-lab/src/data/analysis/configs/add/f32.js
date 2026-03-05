// src/data/analysis/configs/add/f32.js
import metrics from '../../metrics/add_f32.json';

export const f32Config = {
  id: "f32",
  name: "Standard FP32 Addition",
  tag: "Baseline",
  description: "Standard single-precision floating point implementation.",
  
  algorithm: {
    title: "Full-Precision Scalar Addition",
    logic: "32-bit float 연산을 수행하며, 가장 높은 수치적 정밀도를 유지합니다. 정렬 조건이 까다롭지 않아 범용적으로 사용됩니다.",
    strategy: [
      "각 스레드는 32-bit (float) 요소 처리",
      "표준 FP32 연산자(+) 사용",
      "Memory Alignment: 4-byte boundary"
    ]
  },

  blueprint: {
    mem_access: "32-bit (Scalar)",
    instruction: "fadd.f32",
    vector_width: 1,
    alignment_req: "4-byte",
    code_snippet: "out[i] = a[i] + b[i]; // Standard float32 addition"
  },

  metrics: metrics,
  features: ["FP32 Precision", "High Accuracy", "Universal Compatibility"],
  insights: [
    "정밀도가 중요할 때 사용하지만, 메모리 대역폭 소모가 FP16 대비 2배입니다.",
    "Compute-bound 보다는 주로 Memory-bound에 먼저 도달하는 경향이 있습니다.",
    "특별한 최적화 기법 없이도 GPU 최대 성능의 기준점(Baseline)을 제공합니다."
  ]
};