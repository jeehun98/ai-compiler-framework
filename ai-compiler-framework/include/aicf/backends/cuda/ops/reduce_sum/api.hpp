#pragma once
#include <cuda_runtime.h>
#include <aicf/backends/cuda/registry/status.hpp>

namespace aicf::cuda {
Status reduce_sum_lastdim_f32(const float* dY, float* dB, int M, int N, cudaStream_t stream);
} // namespace aicf::cuda
