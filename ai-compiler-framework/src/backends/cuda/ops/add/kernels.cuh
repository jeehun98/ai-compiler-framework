#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// NOTE:
// - 이 파일은 "op 내부(private)" 헤더로 두고, 여기서는 cuda_fp16 include 허용.
// - public API 헤더(api.hpp)는 계속 cuda_fp16 없이 void* 유지해도 됨.

namespace aicf::cuda::add_impl {

__global__ void add_f32_kernel(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ out,
                              int N);

__global__ void add_f16_kernel(const __half* __restrict__ a,
                              const __half* __restrict__ b,
                              __half* __restrict__ out,
                              int N);

__global__ void add_f16x2_kernel(const __half2* __restrict__ a,
                                 const __half2* __restrict__ b,
                                 __half2* __restrict__ out,
                                 int N2); // N2 = N/2

} // namespace aicf::cuda::add_impl
