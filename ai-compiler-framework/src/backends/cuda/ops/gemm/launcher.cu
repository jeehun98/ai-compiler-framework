#include <aicf/backends/cuda/ops/gemm/api.hpp>
#include <aicf/core/log.hpp>
#include "kernels.cuh"

#include <cuda_runtime.h>

namespace aicf::cuda {

static inline cudaStream_t to_cuda_stream(Stream s) {
    return reinterpret_cast<cudaStream_t>(s.handle);
}

Status gemm_f32(const float* A,
                const float* B,
                float* C,
                int M, int N, int K,
                Stream stream) {
    if (!A || !B || !C) return Status::InvalidArgument;
    if (M <= 0 || N <= 0 || K <= 0) return Status::InvalidArgument;

    cudaStream_t cu_stream = to_cuda_stream(stream);

    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              1);

    kernels::gemm_f32_naive_kernel<<<grid, block, 0, cu_stream>>>(A, B, C, M, N, K);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        aicf::log_error("gemm_f32 launch failed: %s", cudaGetErrorString(e));
        return Status::Error;
    }
    return Status::Ok;
}

} // namespace aicf::cuda
