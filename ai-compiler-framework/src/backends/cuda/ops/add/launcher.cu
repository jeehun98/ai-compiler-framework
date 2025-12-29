#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API
#include <aicf/backends/cuda/ops/add/api.hpp>

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
namespace add_impl {

__global__ void add_f32_kernel(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ out,
                               int N) {
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N) out[i] = a[i] + b[i];
}

__global__ void add_f16_kernel(const __half* __restrict__ a,
                               const __half* __restrict__ b,
                               __half* __restrict__ out,
                               int N) {
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N) out[i] = __hadd(a[i], b[i]);
}

__global__ void add_f16x2_kernel(const __half2* __restrict__ a,
                                 const __half2* __restrict__ b,
                                 __half2* __restrict__ out,
                                 int N2) {
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N2) out[i] = __hadd2(a[i], b[i]);
}

}  // namespace add_impl

// -------------------------
// helpers (ND)
// -------------------------
static inline int64_t numel(const TensorDesc& d) {
  int64_t n = 1;
  for (int i = 0; i < d.r.rank; ++i) n *= (int64_t)d.shape[i];
  return n;
}

static inline bool same_shape(const TensorDesc& a, const TensorDesc& b) {
  if (a.r.rank != b.r.rank) return false;
  for (int i = 0; i < a.r.rank; ++i) {
    if (a.shape[i] != b.shape[i]) return false;
  }
  return true;
}

static inline bool is_contig(const TensorDesc& d) { return d.contiguous; }

// -------------------------
// public API implementation
// -------------------------
aicf::Status add_f32(const float* a,
                     const float* b,
                     float* out,
                     int N,
                     aicf::Stream stream) {
  if (!a || !b || !out || N <= 0) return aicf::Status::InvalidArgument;

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  constexpr int kThreads = 256;
  const int blocks = (N + kThreads - 1) / kThreads;

  add_impl::add_f32_kernel<<<blocks, kThreads, 0, s>>>(a, b, out, N);
  return aicf::cuda::shim::cuda_last_error_to_status();
}

// v0.2: keep header free from cuda_fp16 by using void* API
// v0.2+: add half2 fastpath when (N even) && (ptr 4B aligned)
aicf::Status add_f16(const void* a,
                     const void* b,
                     void* out,
                     int N,
                     aicf::Stream stream) {
  if (!a || !b || !out || N <= 0) return aicf::Status::InvalidArgument;

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  constexpr int kThreads = 256;

  const bool even = ((N & 1) == 0);
  const bool a_aligned = (((uintptr_t)a & 0x3u) == 0);
  const bool b_aligned = (((uintptr_t)b & 0x3u) == 0);
  const bool o_aligned = (((uintptr_t)out & 0x3u) == 0);

  if (even && a_aligned && b_aligned && o_aligned) {
    const int N2 = N / 2;
    const int blocks = (N2 + kThreads - 1) / kThreads;

    add_impl::add_f16x2_kernel<<<blocks, kThreads, 0, s>>>(
        (const __half2*)a, (const __half2*)b, (__half2*)out, N2);
  } else {
    const int blocks = (N + kThreads - 1) / kThreads;

    add_impl::add_f16_kernel<<<blocks, kThreads, 0, s>>>(
        (const __half*)a, (const __half*)b, (__half*)out, N);
  }

  return aicf::cuda::shim::cuda_last_error_to_status();
}

// -------------------------
// Registry Variants - v0.2 Plan A (no workspace, no attr semantics yet)
//
// Contract (updated):
//   inputs[0]=A [*], inputs[1]=B [*], outputs[0]=O [*]
//   - contiguous required
//   - shapes must match (same rank + dims)
//   - supports ND by flattening to numel()
//   - out may alias inputs (in-place) by design
// -------------------------

static size_t add_variant_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

static inline bool add_nd_common_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    DType dt) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& O = outputs[0];

  if (A.dtype != dt || B.dtype != dt || O.dtype != dt) return false;
  if (!is_contig(A) || !is_contig(B) || !is_contig(O)) return false;

  if (!same_shape(A, B)) return false;
  if (!same_shape(A, O)) return false;

  return (numel(O) > 0);
}

// ---- F32 variant ----
static bool add_f32_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  return add_nd_common_check(inputs, num_inputs, outputs, num_outputs, DType::kF32);
}

static aicf::Status add_f32_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!add_nd_common_check(inputs, num_inputs, outputs, num_outputs, DType::kF32)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  const int64_t N64 = numel(O);
  const int N = (int)N64;

  constexpr int kThreads = 256;
  const int blocks = (int)((N64 + kThreads - 1) / kThreads);

  add_impl::add_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)A.data,
      (const float*)B.data,
      (float*)O.data,
      N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_add_f32_variant() {
  KernelVariant v{};
  v.name = "add_f32_naive";
  v.priority = 0;
  v.flags = 0;
  v.launch = add_f32_variant_launch;
  v.supported = add_f32_variant_supported;
  v.query_workspace = add_variant_workspace;
  return v;
}

// ---- F16 naive variant ----
static bool add_f16_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  return add_nd_common_check(inputs, num_inputs, outputs, num_outputs, DType::kF16);
}

static aicf::Status add_f16_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!add_nd_common_check(inputs, num_inputs, outputs, num_outputs, DType::kF16)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  const int64_t N64 = numel(O);
  const int N = (int)N64;

  constexpr int kThreads = 256;
  const int blocks = (int)((N64 + kThreads - 1) / kThreads);

  add_impl::add_f16_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half*)A.data,
      (const __half*)B.data,
      (__half*)O.data,
      N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_add_f16_variant() {
  KernelVariant v{};
  v.name = "add_f16_naive";
  v.priority = 0;
  v.flags = 0;
  v.launch = add_f16_variant_launch;
  v.supported = add_f16_variant_supported;
  v.query_workspace = add_variant_workspace;
  return v;
}

// ---- F16 vec2 (half2) variant ----
static inline bool add_f16_vec2_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!add_nd_common_check(inputs, num_inputs, outputs, num_outputs, DType::kF16)) {
    return false;
  }

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& O = outputs[0];

  const int64_t N = numel(O);
  if ((N & 1) != 0) return false;

  constexpr size_t kHalf2Align = 4;
  if (!aicf::cuda::shim::is_aligned_data(A, kHalf2Align)) return false;
  if (!aicf::cuda::shim::is_aligned_data(B, kHalf2Align)) return false;
  if (!aicf::cuda::shim::is_aligned_data(O, kHalf2Align)) return false;

  return true;
}

static bool add_f16_vec2_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  return add_f16_vec2_check(inputs, num_inputs, outputs, num_outputs);
}

static aicf::Status add_f16_vec2_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!add_f16_vec2_check(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  const int64_t N = numel(O);
  const int N2 = (int)(N / 2);

  constexpr int kThreads = 256;
  const int blocks = (int)((N2 + kThreads - 1) / kThreads);

  add_impl::add_f16x2_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half2*)A.data,
      (const __half2*)B.data,
      (__half2*)O.data,
      N2);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_add_f16_vec2_variant() {
  KernelVariant v{};
  v.name = "add_f16_vec2_half2";
  v.priority = 10;
  v.flags = 0;
  v.launch = add_f16_vec2_variant_launch;
  v.supported = add_f16_vec2_variant_supported;
  v.query_workspace = add_variant_workspace;
  return v;
}

}  // namespace aicf::cuda
