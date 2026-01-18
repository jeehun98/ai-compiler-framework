// ============================================================================
// src/backends/cuda/ops/relu/launcher.cu  (core-free / AttrBlob ABI-friendly)
// - no attrs needed (attr ignored)
// - supports contig ANY-RANK via numel flatten
// - f32 naive, f16 naive, f16 half2 fastpath (numel even + 4B aligned)
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cinttypes> // uintptr_t
#include <climits>   // INT_MAX

// public API (optional)
#include <aicf/backends/cuda/ops/relu/api.hpp>

// registry glue
#include <aicf/backends/cuda/registry/status.hpp>
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_blob.hpp>

#include "kernels.cuh"

namespace aicf::cuda {

// ---- cuda error -> Status (core-free) ----
static inline Status cuda_to_status(cudaError_t e) {
  return (e == cudaSuccess) ? Status::Ok : Status::Internal;
}
static inline Status cuda_last_status() {
  return cuda_to_status(cudaGetLastError());
}
static inline bool is_aligned_ptr(const void* p, size_t align) {
  return ((uintptr_t)p % (uintptr_t)align) == 0;
}

// -------------------------
// kernels (definitions live here)
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
    const __half z = __float2half(0.0f);
    out[i] = __hgt(x, z) ? x : z;
  }
}

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
// helpers (ANY-RANK)
// -------------------------
namespace {

static inline int64_t numel_of(const TensorDesc& T) {
  int64_t n = 1;
  const int r = T.rank();
  for (int i = 0; i < r; ++i) n *= T.shape[i];
  return n;
}

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

static size_t relu_workspace(const TensorDesc*, int, const void*) { return 0; }

} // anonymous namespace

// -------------------------
// Variant: F32
// -------------------------
static inline bool relu_f32_check(
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
  return (N > 0 && N <= (int64_t)INT_MAX);
}

static bool relu_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  return relu_f32_check(inputs, num_inputs, outputs, num_outputs);
}

static Status relu_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void*, size_t,
    cudaStream_t stream) {

  if (!relu_f32_check(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& I = inputs[0];
  TensorDesc& O = outputs[0];

  const int N = (int)numel_of(O);

  constexpr int kThreads = 256;
  const int blocks = (N + kThreads - 1) / kThreads;

  cudaGetLastError(); // clear
  relu_impl::relu_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)I.data, (float*)O.data, N);

  return cuda_last_status();
}

KernelVariant make_relu_f32_variant() {
  KernelVariant v{};
  v.name = "relu_f32_naive";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = 0;
  v.launch = relu_f32_launch;
  v.supported = relu_f32_supported;
  v.query_workspace = relu_workspace;
  return v;
}

// -------------------------
// Variant: F16 naive
// -------------------------
static inline bool relu_f16_check(
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
  return (N > 0 && N <= (int64_t)INT_MAX);
}

static bool relu_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  return relu_f16_check(inputs, num_inputs, outputs, num_outputs);
}

static Status relu_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void*, size_t,
    cudaStream_t stream) {

  if (!relu_f16_check(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& I = inputs[0];
  TensorDesc& O = outputs[0];

  const int N = (int)numel_of(O);

  constexpr int kThreads = 256;
  const int blocks = (N + kThreads - 1) / kThreads;

  cudaGetLastError(); // clear
  relu_impl::relu_f16_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half*)I.data, (__half*)O.data, N);

  return cuda_last_status();
}

KernelVariant make_relu_f16_variant() {
  KernelVariant v{};
  v.name = "relu_f16_naive";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = 0;
  v.launch = relu_f16_launch;
  v.supported = relu_f16_supported;
  v.query_workspace = relu_workspace;
  return v;
}

// -------------------------
// Variant: F16 half2 (vec2)
// -------------------------
static inline bool relu_f16_vec2_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!relu_f16_check(inputs, num_inputs, outputs, num_outputs)) return false;

  const int64_t N = numel_of(outputs[0]);
  if ((N & 1) != 0) return false;

  constexpr size_t kAlign = 4;
  if (!is_aligned_ptr(inputs[0].data, kAlign)) return false;
  if (!is_aligned_ptr(outputs[0].data, kAlign)) return false;

  return true;
}

static bool relu_f16_vec2_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  return relu_f16_vec2_check(inputs, num_inputs, outputs, num_outputs);
}

static Status relu_f16_vec2_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void*, size_t,
    cudaStream_t stream) {

  if (!relu_f16_vec2_check(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& I = inputs[0];
  TensorDesc& O = outputs[0];

  const int N = (int)numel_of(O);
  const int N2 = N / 2;

  constexpr int kThreads = 256;
  const int blocks = (N2 + kThreads - 1) / kThreads;

  cudaGetLastError(); // clear
  relu_impl::relu_f16x2_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half2*)I.data, (__half2*)O.data, N2);

  return cuda_last_status();
}

KernelVariant make_relu_f16_vec2_variant() {
  KernelVariant v{};
  v.name = "relu_f16_vec2_half2";
  v.priority = 10;
  v.flags = 0;
  v.expected_attr_schema_id = 0;
  v.launch = relu_f16_vec2_launch;
  v.supported = relu_f16_vec2_supported;
  v.query_workspace = relu_workspace;
  return v;
}

} // namespace aicf::cuda
