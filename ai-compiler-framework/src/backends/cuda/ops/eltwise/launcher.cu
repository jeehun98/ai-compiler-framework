#include "kernels.cuh"
#include <aicf/backends/cuda/ops/eltwise/api.hpp>
#include <aicf/backends/cuda/nvtx.hpp>

namespace aicf::cuda::ops::eltwise {

__global__ void add_f32_kernel(
    float* out,
    const float* a,
    const float* b,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void relu_f32_kernel(
    float* out,
    const float* x,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i] = v > 0.f ? v : 0.f;
    }
}

void add_f32(
    float* out,
    const float* a,
    const float* b,
    int64_t n,
    cudaStream_t stream
) {
    //AICF_NVTX_RANGE("aicf::eltwise::add_f32");

    constexpr int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    add_f32_kernel<<<blocks, threads, 0, stream>>>(out, a, b, n);
}

void relu_f32(
    float* out,
    const float* x,
    int64_t n,
    cudaStream_t stream
) {
    //AICF_NVTX_RANGE("aicf::eltwise::relu_f32");

    constexpr int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    relu_f32_kernel<<<blocks, threads, 0, stream>>>(out, x, n);
}

} // namespace
