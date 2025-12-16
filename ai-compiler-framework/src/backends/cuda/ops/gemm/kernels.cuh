#pragma once
#include <cuda_runtime.h>

namespace aicf::cuda::kernels {

// naive GEMM: each thread computes one C element
__global__ void gemm_f32_naive_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K) {
    int row = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    int col = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    int a_base = row * K;
    int b_base = col; // B[k*N + col]
    for (int k = 0; k < K; ++k) {
        acc += A[a_base + k] * B[k * N + b_base];
    }
    C[row * N + col] = acc;
}

} // namespace aicf::cuda::kernels
