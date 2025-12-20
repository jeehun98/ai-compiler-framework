#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
    out[i] = __hgt(x, __float2half(0.0f)) ? x : __float2half(0.0f);
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

aicf::Status relu_f16(const void* in,
                      void* out,
                      int N,
                      aicf::Stream stream) {
  if (!in || !out || N <= 0) return aicf::Status::InvalidArgument;

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  constexpr int kThreads = 256;
  const int blocks = (N + kThreads - 1) / kThreads;
  relu_impl::relu_f16_kernel<<<blocks, kThreads, 0, s>>>(
      (const __half*)in, (__half*)out, N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

// -------------------------
// Registry Variants - v0.2 Plan A (no workspace, no attr semantics yet)
//
// Contract:
//   inputs[0]=I [N], outputs[0]=O [N]
//   contig + rank1; dtype differs per variant
// -------------------------

// ---- F32 variant ----
static inline bool relu_f32_variant_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& I = inputs[0];
  const TensorDesc& O = outputs[0];

  if (!aicf::cuda::shim::is_f32_contig_1d(I)) return false;
  if (!aicf::cuda::shim::is_f32_contig_1d(O)) return false;
  if (!aicf::cuda::shim::same_shape_1d(I, O)) return false;
  if (O.shape[0] <= 0) return false;

  return true;
}

static bool relu_f32_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  if (!inputs || !outputs) return false;
  return relu_f32_variant_check(inputs, num_inputs, outputs, num_outputs);
}

static aicf::Status relu_f32_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;
  if (!relu_f32_variant_check(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& I = inputs[0];
  TensorDesc& O = outputs[0];

  const int N = static_cast<int>(O.shape[0]);

  constexpr int kThreads = 256;
  const int blocks = (N + kThreads - 1) / kThreads;
  relu_impl::relu_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)I.data,
      (float*)O.data,
      N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

// ---- F16 variant ----
static inline bool relu_f16_variant_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& I = inputs[0];
  const TensorDesc& O = outputs[0];

  if (!aicf::cuda::shim::is_f16_contig_1d(I)) return false;
  if (!aicf::cuda::shim::is_f16_contig_1d(O)) return false;
  if (!aicf::cuda::shim::same_shape_1d(I, O)) return false;
  if (O.shape[0] <= 0) return false;

  return true;
}

static bool relu_f16_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  if (!inputs || !outputs) return false;
  return relu_f16_variant_check(inputs, num_inputs, outputs, num_outputs);
}

static aicf::Status relu_f16_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;
  if (!relu_f16_variant_check(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& I = inputs[0];
  TensorDesc& O = outputs[0];

  const int N = static_cast<int>(O.shape[0]);

  constexpr int kThreads = 256;
  const int blocks = (N + kThreads - 1) / kThreads;
  relu_impl::relu_f16_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half*)I.data,
      (__half*)O.data,
      N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

static size_t relu_variant_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

// factories
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

} // namespace aicf::cuda
