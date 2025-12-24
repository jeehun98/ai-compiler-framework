#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace aicf::cuda::sgd_step_impl {

// -------------------------
// f32 scalar
// param -= lr * grad
// -------------------------
__global__ void sgd_step_f32_kernel(float* __restrict__ param,
                                   const float* __restrict__ grad,
                                   int64_t numel,
                                   float lr) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (i >= numel) return;
  param[i] = param[i] - lr * grad[i];
}

// -------------------------
// f16 scalar
// param -= lr * grad
// -------------------------
__global__ void sgd_step_f16_kernel(__half* __restrict__ param,
                                   const __half* __restrict__ grad,
                                   int64_t numel,
                                   float lr) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (i >= numel) return;

  float p = __half2float(param[i]);
  float g = __half2float(grad[i]);
  p -= lr * g;
  param[i] = __float2half(p);
}

// -------------------------
// f16 half2 vectorized
// param2 -= lr * grad2  (2 elems per thread item)
// requirements handled in supported():
//  - numel even
//  - 4B alignment for param/grad/out pointers
// -------------------------
__global__ void sgd_step_f16_half2_kernel(__half2* __restrict__ param2,
                                         const __half2* __restrict__ grad2,
                                         int64_t numel2,   // number of half2 elements
                                         float lr) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (i >= numel2) return;

  __half2 p2 = param2[i];
  __half2 g2 = grad2[i];

  float2 pf = __half22float2(p2);
  float2 gf = __half22float2(g2);

  pf.x -= lr * gf.x;
  pf.y -= lr * gf.y;

  param2[i] = __floats2half2_rn(pf.x, pf.y);
}

} // namespace aicf::cuda::sgd_step_impl
