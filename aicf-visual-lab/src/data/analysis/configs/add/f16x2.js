// src/data/analysis/configs/add/f16x2.js
import metrics from '../../metrics/add_f16x2.json';

export const f16x2Config = {
  id: "f16x2",
  name: "Vectorized FP16 (Half2)",
  tag: "Fast Path",
  description: "Highly optimized half-precision path using SIMD vectorization.",
  
  algorithm: {
    title: "SIMD Half2 Vectorization",
    logic: "4-byte 정렬된 포인터를 활용하여 두 개의 FP16 데이터를 한 번에 읽어옵니다. 단일 명령어로 2개의 연산을 동시 처리하여 처리량을 극대화합니다.",
    strategy: [
      "32-bit (2 x FP16) 벡터 로드 수행",
      "CUDA Intrinsic: __hadd2() SIMD 연산",
      "조건: N이 짝수여야 하며 모든 포인터가 4-byte 정렬되어야 함"
    ]
  },

  blueprint: {
    mem_access: "32-bit (Vectorized)",
    instruction: "__hadd2(a2, b2)",
    vector_width: 2,
    alignment_req: "4-byte (Strict)",
    code_snippet: "out2[i] = __hadd2(a2[i], b2[i]); // SIMD vectorized addition"
  },

  metrics: metrics,
  features: ["SIMD Optimized", "Reduced Instruction Count", "Max Bandwidth"],
  insights: [
    "Instruction Issue rate를 절반으로 줄여 SM 효율성을 극대화했습니다.",
    "Naive FP16 대비 동일 시간당 2배의 연산을 처리할 수 있는 구조입니다.",
    "메모리 정렬이 깨질 경우 Naive 경로로 fallback 되므로 정렬 관리가 핵심입니다."
  ]
};