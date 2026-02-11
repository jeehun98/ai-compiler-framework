// src/data/gemm.js
export const gemmData = {
  id: "GEMM",
  category: "Linear Layers",

  semantic: {
    formula: "C = \\alpha(A \\times B) + \\beta C",
    description:
      "행렬 곱셈의 수학적 본질은 선형 변환의 합성입니다. AICF는 결과 행렬의 각 원소가 독립적으로 계산될 수 있다는 '병렬적 의미'에 집중합니다.",
    decomposition: [
      "Blocking: 거대한 행렬을 GPU L1/L2 캐시에 적합한 Tile 단위로 분할",
      "K-Loop Unrolling: 연산 밀도를 높이기 위해 내부 루프 전개",
      "Epilogue: 계산 직후 메모리 이동 없이 Activation(ReLU 등) 병합",
    ],
    precision_policy: "FP32 (Relative Error < 1e-7 허용)",
  },

  optimization: {
    strategy: "2D Hierarchical Tiling",
    details: [
      "Shared Memory Bank Conflict 회피를 위한 Padding 적용",
      "데이터 재사용률 극대화를 위한 Register Blocking (8x8)",
      "Warp-level Matrix Multiply (WMMA) 구조 모방 구현",
    ],
    memory_reuse: "14.2x (Global Mem Access 대비)",
    throughput: "84.2 TFLOPS",
    occupancy: 92,
  },

  // ✅ App.jsx가 기대하는 경로: data.performance.latency.{pytorch, torch_compile, ours}
  performance: {
    latency: {
      pytorch: 210,
      torch_compile: 155,
      ours: 120,
    },
  },

  cudaCode: `// Optimized Tiled GEMM Kernel
__global__ void gemm_optimized(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sA[TILE_K][TILE_M];
    __shared__ float sB[TILE_K][TILE_N];

    // 1. Thread-local accumulation in Registers
    float res[8][8] = {0.0f};

    // 2. Main K-loop with Prefetching
    for (int k = 0; k < K; k += TILE_K) {
        load_tiles_to_shared(sA, sB);
        __syncthreads();
        compute_outer_product(res, sA, sB);
        __syncthreads();
    }
}`,
};
