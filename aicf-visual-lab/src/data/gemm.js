export const gemmData = {
  id: "GEMM",
  category: "Linear Transform / Projection",
  // 1. 의미론적 본질 (Mathematical & Geometric)
  semantic: {
    formula: "C = \\alpha(A \\times B) + \\beta C",
    concept: "Semantic Projection & Hypothesis Testing",
    description: "데이터 샘플(A)이 가설 집합(B)과 얼마나 정렬되는지 평가하고, 과거 상태(C)와 병합하는 과정",
    // ✅ 추가
    decomposition: [
      "Blocking: Tile 단위 분할",
      "K-Loop Unrolling: 연산 밀도 상승",
      "Epilogue: Bias/Act 등 병합",
    ],
    // 차원 의미 (K차원의 소거)
    dimensions: {
      M: "Data Samples",
      K: "Semantic Search Space (Hypothesis Count)",
      N: "Feature Extraction Channels"
    },

    // 의미 보존 속성 (Semantic Attributes)
    attributes: [
      { label: "Rank Tolerance", value: "0.05", desc: "Low-rank approximation 허용 오차" },
      { label: "Order Preserve", value: "Required", desc: "Top-K 순위 보존 필수 (Softmax 민감도)" },
      { label: "Energy Preserve", value: "99.9%", desc: "SVD 분해 시 정보 보존 임계치" }
    ],

    // 최적화 규칙 (Semantic Rules)
    rules: [
      "Rule 6.5: Semantic Anchor Fusion (Bias+Norm+Act)",
      "Rule 6.6: Hypothesis Pruning (기여도 낮은 K-axis 제거)",
      "Rule 8.1: Rank Invariance (순위 보존 하에 극단적 양자화)"
    ]
  },

  // 2. 커널 최적화 (Physical Mapping)
  optimization: {
    strategy: "2D Hierarchical Tiling",
    memory_reuse: "14.2x",
    throughput: "84.2 TFLOPS",
    occupancy: 92,
    details: [
      "K-Loop Unrolling for Hypothesis Testing Speed",
      "Shared Memory Bank Conflict Avoidance",
      "Epilogue Fusion (State Merge Logic)"
    ]
  },

  performance: {
    latency: { ours: 120, pytorch: 210, torch_compile: 155 },
  },

  cudaCode: `// AICF Generated: Semantic-fused GEMM
__global__ void gemm_semantic_fused(...) {
    // 1. Register Blocking for K-axis (Search Space)
    // 2. Shared Memory Tiling (Data Reuse)
    // 3. Epilogue Fusion (State Merge: α, β)
}`
};