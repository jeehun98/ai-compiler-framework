#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace aicf::cuda::step_inc_impl {

// step[i] += 1 (numel>=1). scalar(0d) is treated as numel=1 in launcher.
__global__ void step_inc_i32_kernel(int32_t* __restrict__ step, int64_t numel);

} // namespace aicf::cuda::step_inc_impl
