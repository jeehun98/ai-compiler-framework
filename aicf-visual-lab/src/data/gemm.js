// src/data/gemm.js
export const gemmData = {
  id: "GEMM",
  category: "Linear Layers",
  semantic: {
    formula: "C = \\alpha(A \\times B) + \\beta C",
    decomposition: ["Tile Block Division", "Accumulation Loop", "Epilogue Fusion"],
    precision: "FP32",
  },
  optimization: {
    strategy: "Shared Memory Tiling",
    details: ["Bank Conflict Avoidance", "Register Blocking"],
    memory_reuse: "14.2x",
    throughput: "84.2 TFLOPS",
    occupancy: 92
  },
  performance: {
    latency: { ours: 120, pytorch: 210, torch_compile: 155 },
  },
  cudaCode: `__global__ void gemm_kernel(...) {
  __shared__ float tile[32][32];
  // 1. Load A, B Tiles to Shared Mem
  // 2. Compute Outer Product
  // 3. Store result with Bias/Act
}`
};