#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace aicf::cuda::biascorr_impl {

// step: int32 scalar (S[0])
// outputs: bc1_inv[0], bc2_inv[0]
__global__ void biascorr_kernel(const int32_t* step,
                               float beta1, float beta2,
                               float* bc1_inv, float* bc2_inv);

} // namespace aicf::cuda::biascorr_impl
