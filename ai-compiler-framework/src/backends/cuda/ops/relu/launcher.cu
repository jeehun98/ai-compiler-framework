#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API
#include <aicf/backends/cuda/ops/relu/api.hpp>

// registry glue
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>

// common shim
#include "aicf/backends/cuda/ops/_common/shim/launch.hpp"
#include "aicf/backends/cuda/ops/_common/shim/status.hpp"
#include "aicf/backends/cuda/ops/_common/shim/validate.hpp"

#include "kernels.cuh"

namespace aicf::cuda {

// -------------------------
// kernels
// -------------------------
namespace relu_impl {

__global__ void relu_f32_kernel(const float* __restrict__ in,
                                float* __restrict__ out,
                                int N) {
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N) {
    const float x = in[i];
    out[i] = (x > 0.0f) ? x : 0.0f;
  }
}

__global__ void relu_f16_kernel(const __half* __restrict__ in,
                                __half* __restrict__ out,
                                int N) {
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N) {
    const __half x = in[i];
    // out = max(x, 0)
    out[i] = __hgt(x, __float2half(0.0f)) ? x : __float2half(0.0f);
  }
}

// half2 vectorized: out = max(in, 0) lane-wise
__global__ void relu_f16x2_kernel(const __half2* __restrict__ in,
                                  __half2* __restrict__ out,
                                  int N2) {
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N2) {
    const __half2 x = in[i];
    const __half2 z = __float2half2_rn(0.0f);
    out[i] = __hmax2(x, z);
  }
}

} // namespace relu_impl

// -------------------------
// public API implementation
// -------------------------
aicf::Status relu_f32(const float* in,
                      float* out,
                      int N,
                      aicf::Stream stream) {
  if (!in || !out || N <= 0) return aicf::Status::InvalidArgument;

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  constexpr int kThreads = 256;
  const int blocks = (N + kThreads - 1) / kThreads;
  relu_impl::relu_f32_kernel<<<blocks, kThreads, 0, s>>>(in, out, N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

// v0.2: keep header free from cuda_fp16 by using void* API
// v0.2+: add half2 fastpath when (N even) && (ptr 4B aligned)
aicf::Status relu_f16(const void* in,
                      void* out,
                      int N,
                      aicf::Stream stream) {
  if (!in || !out || N <= 0) return aicf::Status::InvalidArgument;

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  constexpr int kThreads = 256;

  const bool even = ((N & 1) == 0);
  const bool i_aligned = (((uintptr_t)in & 0x3u) == 0);
  const bool o_aligned = (((uintptr_t)out & 0x3u) == 0);

  if (even && i_aligned && o_aligned) {
    const int N2 = N / 2;
    const int blocks = (N2 + kThreads - 1) / kThreads;
    relu_impl::relu_f16x2_kernel<<<blocks, kThreads, 0, s>>>(
        (const __half2*)in, (__half2*)out, N2);
  } else {
    const int blocks = (N + kThreads - 1) / kThreads;
    relu_impl::relu_f16_kernel<<<blocks, kThreads, 0, s>>>(
        (const __half*)in, (__half*)out, N);
  }

  return aicf::cuda::shim::cuda_last_error_to_status();
}

// -------------------------
// Registry Variants
//
// CHANGED:
//   - Previously: only contig + rank1 ([N]) using is_*_contig_1d()
//   - Now: contig ANY-RANK, use numel as N
//
// Contract:
//   inputs[0]=I [*], outputs[0]=O [*]
//   - same rank + same shape
//   - contiguous row-major
//   - dtype differs per variant
// -------------------------

namespace {

// numel for any rank
static inline int64_t numel_of(const TensorDesc& T) {
  int64_t n = 1;
  const int r = T.rank();
  for (int i = 0; i < r; ++i) n *= T.shape[i];
  return n;
}

// check row-major contiguous for any rank
static inline bool is_contig_anyrank(const TensorDesc& T) {
  const int r = T.rank();
  if (r <= 0) return false;
  if (T.stride[r - 1] != 1) return false;
  for (int i = r - 2; i >= 0; --i) {
    if (T.stride[i] != T.shape[i + 1] * T.stride[i + 1]) return false;
  }
  return true;
}

static inline bool same_shape_anyrank(const TensorDesc& A, const TensorDesc& B) {
  if (A.rank() != B.rank()) return false;
  for (int i = 0; i < A.rank(); ++i) {
    if (A.shape[i] != B.shape[i]) return false;
  }
  return true;
}

static size_t relu_variant_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

} // anonymous namespace

// ---- F32 variant ----
static inline bool relu_f32_variant_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& I = inputs[0];
  const TensorDesc& O = outputs[0];

  if (I.dtype != DType::kF32) return false;
  if (O.dtype != DType::kF32) return false;

  if (!same_shape_anyrank(I, O)) return false;
  if (!is_contig_anyrank(I) || !is_contig_anyrank(O)) return false;

  const int64_t N = numel_of(O);
  return (N > 0 && N <= INT_MAX);
}

static bool relu_f32_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return relu_f32_variant_check(inputs, num_inputs, outputs, num_outputs);
}

static aicf::Status relu_f32_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!relu_f32_variant_check(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& I = inputs[0];
  TensorDesc& O = outputs[0];

  const int N = (int)numel_of(O);

  constexpr int kThreads = 256;
  const int blocks = (N + kThreads - 1) / kThreads;

  relu_impl::relu_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)I.data, (float*)O.data, N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_relu_f32_variant() {
  KernelVariant v{};
  v.name = "relu_f32_naive";
  v.priority = 0;
  v.flags = 0;
  v.launch = relu_f32_variant_launch;
  v.supported = relu_f32_variant_supported;
  v.query_workspace = relu_variant_workspace;
  return v;
}

// ---- F16 naive variant ----
static inline bool relu_f16_variant_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& I = inputs[0];
  const TensorDesc& O = outputs[0];

  if (I.dtype != DType::kF16) return false;
  if (O.dtype != DType::kF16) return false;

  if (!same_shape_anyrank(I, O)) return false;
  if (!is_contig_anyrank(I) || !is_contig_anyrank(O)) return false;

  const int64_t N = numel_of(O);
  return (N > 0 && N <= INT_MAX);
}

static bool relu_f16_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return relu_f16_variant_check(inputs, num_inputs, outputs, num_outputs);
}

static aicf::Status relu_f16_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!relu_f16_variant_check(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& I = inputs[0];
  TensorDesc& O = outputs[0];

  const int N = (int)numel_of(O);

  constexpr int kThreads = 256;
  const int blocks = (N + kThreads - 1) / kThreads;

  relu_impl::relu_f16_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half*)I.data, (__half*)O.data, N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_relu_f16_variant() {
  KernelVariant v{};
  v.name = "relu_f16_naive";
  v.priority = 0;
  v.flags = 0;
  v.launch = relu_f16_variant_launch;
  v.supported = relu_f16_variant_supported;
  v.query_workspace = relu_variant_workspace;
  return v;
}

// ---- F16 vec2 (half2) variant ----
static inline bool relu_f16_vec2_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!relu_f16_variant_check(inputs, num_inputs, outputs, num_outputs)) {
    return false;
  }

  const TensorDesc& I = inputs[0];
  const TensorDesc& O = outputs[0];

  const int64_t N = numel_of(O);

  // even length
  if ((N & 1) != 0) return false;

  // half2 requires 4B alignment
  constexpr size_t kAlign = 4;
  if (!aicf::cuda::shim::is_aligned_data(I, kAlign)) return false;
  if (!aicf::cuda::shim::is_aligned_data(O, kAlign)) return false;

  return true;
}

static bool relu_f16_vec2_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return relu_f16_vec2_check(inputs, num_inputs, outputs, num_outputs);
}

static aicf::Status relu_f16_vec2_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!relu_f16_vec2_check(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& I = inputs[0];
  TensorDesc& O = outputs[0];

  const int N = (int)numel_of(O);
  const int N2 = N / 2;

  constexpr int kThreads = 256;
  const int blocks = (N2 + kThreads - 1) / kThreads;

  relu_impl::relu_f16x2_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half2*)I.data, (__half2*)O.data, N2);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_relu_f16_vec2_variant() {
  KernelVariant v{};
  v.name = "relu_f16_vec2_half2";
  v.priority = 10;  // vec2 wins over naive
  v.flags = 0;
  v.launch = relu_f16_vec2_variant_launch;
  v.supported = relu_f16_vec2_variant_supported;
  v.query_workspace = relu_variant_workspace;
  return v;
}

} // namespace aicf::cuda
