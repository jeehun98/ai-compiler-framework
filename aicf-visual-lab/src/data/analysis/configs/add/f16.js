// src/data/analysis/configs/add/f16.js
import metrics from '../../metrics/add_f16.json';

export const f16Config = {
  id: "f16",
  name: "Naive FP16 Addition",
  tag: "Scalar",
  description: "Standard half-precision implementation without vectorization.",
  
  // 커널 알고리즘 핵심 정보
  algorithm: {
    title: "Scalar FP16 Addition",
    logic: "각 GPU 스레드가 16-bit (__half) 요소를 하나씩 로드하여 연산합니다. 정렬에 구애받지 않으나 대역폭 활용도가 낮습니다.",
    strategy: [
      "각 스레드는 단일 __half 연산 수행",
      "CUDA Intrinsic: __hadd() 사용",
      "Memory Alignment: 2-byte boundary"
    ]
  },

  // 시각적 특징 요약 (Blueprint)
  blueprint: {
    mem_access: "16-bit (Scalar)",
    instruction: "__hadd(a, b)",
    vector_width: 1,
    alignment_req: "2-byte",
    code_snippet: "out[i] = a[i] + b[i]; // Standard half addition"
  },

  metrics: metrics,
  features: ["FP16 Precision", "Scalar Access"],
  insights: [
    "벡터화가 적용되지 않아 Instruction Overhead가 상대적으로 높습니다.",
    "메모리 요청 주소의 정렬 상태가 낮은 경우에도 안정적으로 작동합니다.",
    "L1/L2 캐시 라인 활용도가 f16x2 방식에 비해 비효율적일 수 있습니다."
  ]
};