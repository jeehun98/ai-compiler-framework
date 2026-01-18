#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace aicf::cuda {

// Pout, Mout, Vout can alias P/M/V respectively (in-place allowed)
__global__ void adam_step_f32_kernel_v2(
    float* __restrict__ Pout,
    const float* __restrict__ G,
    const float* __restrict__ M,
    const float* __restrict__ V,
    float* __restrict__ Mout,
    float* __restrict__ Vout,
    int64_t n,
    float lr, float beta1, float beta2, float eps,
    const float* __restrict__ bc1,
    const float* __restrict__ bc2);

} // namespace aicf::cuda
