#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace aicf::cuda::step_inc_impl {

// step: int32 scalar (or tiny tensor) in-place increment
__global__ void step_inc_i32_kernel(int32_t* __restrict__ step, int64_t numel);

} // namespace aicf::cuda::step_inc_impl
