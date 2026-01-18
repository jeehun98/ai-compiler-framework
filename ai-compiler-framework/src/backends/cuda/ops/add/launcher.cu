// ============================================================================
// src/backends/cuda/ops/add/launcher.cu  (core-free / AttrBlob ABI-friendly)
// - no attrs needed (attr ignored)
// - supports ND contiguous tensors by flattening numel()
// - f32 naive, f16 naive, f16 half2 fastpath (numel even + 4B aligned)
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <cinttypes> // uintptr_t

// public API (optional)
#include <aicf/backends/cuda/ops/add/api.hpp>

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

// dtype-generic common check
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

static size_t add_workspace(const TensorDesc*, int, const void*) { return 0; }

// -------------------------
// Variant: F32
// -------------------------
static bool add_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  return add_nd_common_check(inputs, num_inputs, outputs, num_outputs, DType::kF32);
}

static Status add_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void*, size_t,
    cudaStream_t stream) {

  if (!add_nd_common_check(inputs, num_inputs, outputs, num_outputs, DType::kF32)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  const int64_t N64 = numel(O);
  const int N = (int)N64;

  constexpr int kThreads = 256;
  const int blocks = (int)((N64 + kThreads - 1) / kThreads);

  cudaGetLastError(); // clear
  add_impl::add_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)A.data, (const float*)B.data, (float*)O.data, N);

  return cuda_last_status();
}

KernelVariant make_add_f32_variant() {
  KernelVariant v{};
  v.name = "add_f32_naive";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = 0;
  v.launch = add_f32_launch;
  v.supported = add_f32_supported;
  v.query_workspace = add_workspace;
  return v;
}

// -------------------------
// Variant: F16 naive
// -------------------------
static bool add_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  return add_nd_common_check(inputs, num_inputs, outputs, num_outputs, DType::kF16);
}

static Status add_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void*, size_t,
    cudaStream_t stream) {

  if (!add_nd_common_check(inputs, num_inputs, outputs, num_outputs, DType::kF16)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  const int64_t N64 = numel(O);
  const int N = (int)N64;

  constexpr int kThreads = 256;
  const int blocks = (int)((N64 + kThreads - 1) / kThreads);

  cudaGetLastError(); // clear
  add_impl::add_f16_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half*)A.data, (const __half*)B.data, (__half*)O.data, N);

  return cuda_last_status();
}

KernelVariant make_add_f16_variant() {
  KernelVariant v{};
  v.name = "add_f16_naive";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = 0;
  v.launch = add_f16_launch;
  v.supported = add_f16_supported;
  v.query_workspace = add_workspace;
  return v;
}

// -------------------------
// Variant: F16 half2 (vec2)
// -------------------------
static inline bool add_f16_vec2_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!add_nd_common_check(inputs, num_inputs, outputs, num_outputs, DType::kF16)) {
    return false;
  }

  const TensorDesc& O = outputs[0];
  const int64_t N = numel(O);
  if ((N & 1) != 0) return false;

  // 4B align
  constexpr size_t kAlign = 4;
  if (!is_aligned_ptr(inputs[0].data, kAlign)) return false;
  if (!is_aligned_ptr(inputs[1].data, kAlign)) return false;
  if (!is_aligned_ptr(outputs[0].data, kAlign)) return false;

  return true;
}

static bool add_f16_vec2_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  return add_f16_vec2_check(inputs, num_inputs, outputs, num_outputs);
}

static Status add_f16_vec2_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void*, size_t,
    cudaStream_t stream) {

  if (!add_f16_vec2_check(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  const int64_t N = numel(O);
  const int N2 = (int)(N / 2);

  constexpr int kThreads = 256;
  const int blocks = (int)((N2 + kThreads - 1) / kThreads);

  cudaGetLastError(); // clear
  add_impl::add_f16x2_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half2*)A.data, (const __half2*)B.data, (__half2*)O.data, N2);

  return cuda_last_status();
}

KernelVariant make_add_f16_vec2_variant() {
  KernelVariant v{};
  v.name = "add_f16_vec2_half2";
  v.priority = 10;
  v.flags = 0;
  v.expected_attr_schema_id = 0;
  v.launch = add_f16_vec2_launch;
  v.supported = add_f16_vec2_supported;
  v.query_workspace = add_workspace;
  return v;
}

} // namespace aicf::cuda
