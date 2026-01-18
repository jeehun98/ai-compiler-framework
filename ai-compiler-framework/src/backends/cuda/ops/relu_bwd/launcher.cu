// ============================================================================
// src/backends/cuda/ops/relu_bwd/launcher.cu  (core-free / minimal)
// Contract (framework):
//   inputs[0] = dOut
//   inputs[1] = Y
//   outputs[0]= dY
// - contig required, same shape, dtype per variant
// - no attrs, no workspace
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

#include <aicf/backends/cuda/registry/status.hpp>
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>

#include "kernels.cuh"

namespace aicf::cuda {

// ---- cuda error -> Status (core-free) ----
static inline Status cuda_to_status(cudaError_t e) {
  return (e == cudaSuccess) ? Status::Ok : Status::Internal;
}
static inline Status cuda_last_status() {
  return cuda_to_status(cudaGetLastError());
}

// -------------------------
// kernels
// -------------------------
namespace relu_bwd_impl {

__global__ void relu_bwd_f32_kernel(const float* __restrict__ Y,
                                   const float* __restrict__ dOut,
                                   float* __restrict__ dY,
                                   int64_t numel) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (i >= numel) return;
  const float y = Y[i];
  dY[i] = (y > 0.0f) ? dOut[i] : 0.0f;
}

__global__ void relu_bwd_f16_kernel(const __half* __restrict__ Y,
                                   const __half* __restrict__ dOut,
                                   __half* __restrict__ dY,
                                   int64_t numel) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (i >= numel) return;

  const __half y = Y[i];
  dY[i] = __hgt(y, __float2half(0.0f)) ? dOut[i] : __float2half(0.0f);
}

__global__ void relu_bwd_f16x2_kernel(const __half2* __restrict__ Y,
                                     const __half2* __restrict__ dOut,
                                     __half2* __restrict__ dY,
                                     int64_t numel2) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (i >= numel2) return;

  const __half2 y = Y[i];
  const __half2 g = dOut[i];

  const __half y0 = __low2half(y);
  const __half y1 = __high2half(y);
  const __half g0 = __low2half(g);
  const __half g1 = __high2half(g);

  const __half o0 = __hgt(y0, __float2half(0.0f)) ? g0 : __float2half(0.0f);
  const __half o1 = __hgt(y1, __float2half(0.0f)) ? g1 : __float2half(0.0f);

  dY[i] = __halves2half2(o0, o1);
}

} // namespace relu_bwd_impl

// -------------------------
// helpers
// -------------------------
static inline bool is_f32_contig(const TensorDesc& T) {
  return (T.dtype == DType::kF32) && T.contiguous;
}
static inline bool is_f16_contig(const TensorDesc& T) {
  return (T.dtype == DType::kF16) && T.contiguous;
}

static inline bool same_shape(const TensorDesc& A, const TensorDesc& B) {
  if (A.rank() != B.rank()) return false;
  for (int64_t i = 0; i < A.rank(); ++i) {
    if (A.shape[i] != B.shape[i]) return false;
  }
  return true;
}

static inline bool compute_numel(const TensorDesc& T, int64_t* out_numel) {
  if (!out_numel) return false;
  const int64_t r = T.rank();
  if (r < 1) return false;

  int64_t n = 1;
  for (int64_t i = 0; i < r; ++i) {
    const int64_t d = T.shape[i];
    if (d <= 0) return false;
    n *= d;
  }
  if (n <= 0) return false;
  *out_numel = n;
  return true;
}

static inline bool is_aligned_ptr(const void* p, size_t align) {
  return (((uintptr_t)p) & (align - 1)) == 0;
}

// Contract:
// inputs[0]=dOut, inputs[1]=Y, outputs[0]=dY
static inline bool relu_bwd_check_dt(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    bool (*is_ok)(const TensorDesc&)) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& dOut = inputs[0];
  const TensorDesc& Y    = inputs[1];
  const TensorDesc& dY   = outputs[0];

  if (!is_ok(Y) || !is_ok(dOut) || !is_ok(dY)) return false;
  if (Y.rank() < 1) return false;
  if (!same_shape(Y, dOut)) return false;
  if (!same_shape(Y, dY)) return false;

  int64_t numel = 0;
  if (!compute_numel(Y, &numel)) return false;
  return true;
}

static inline bool relu_bwd_f16_vec2_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!relu_bwd_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f16_contig)) {
    return false;
  }

  const TensorDesc& dOut = inputs[0];
  const TensorDesc& Y    = inputs[1];
  const TensorDesc& dY   = outputs[0];

  int64_t numel = 0;
  if (!compute_numel(Y, &numel)) return false;
  if ((numel & 1) != 0) return false;

  constexpr size_t kAlign = 4;
  if (!is_aligned_ptr(Y.data, kAlign)) return false;
  if (!is_aligned_ptr(dOut.data, kAlign)) return false;
  if (!is_aligned_ptr(dY.data, kAlign)) return false;
  return true;
}

static size_t relu_bwd_workspace(const TensorDesc*, int, const void*) { return 0; }

// -------------------------
// Variant: F32
// -------------------------
static bool relu_bwd_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void*) {
  return relu_bwd_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f32_contig);
}

static Status relu_bwd_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void*,
    void*, size_t,
    cudaStream_t stream) {

  if (!relu_bwd_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f32_contig)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& dOut = inputs[0];
  const TensorDesc& Y    = inputs[1];
  TensorDesc& dY         = outputs[0];

  int64_t numel = 0;
  if (!compute_numel(Y, &numel)) return Status::InvalidArgument;

  constexpr int kThreads = 256;
  int64_t blocks64 = (numel + kThreads - 1) / kThreads;
  if (blocks64 < 1) blocks64 = 1;
  if (blocks64 > 65535) blocks64 = 65535;
  const int blocks = (int)blocks64;

  cudaGetLastError(); // clear
  relu_bwd_impl::relu_bwd_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)Y.data, (const float*)dOut.data, (float*)dY.data, numel);

  return cuda_last_status();
}

KernelVariant make_relu_bwd_f32_variant() {
  KernelVariant v{};
  v.name = "relu_bwd_f32";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = 0;
  v.launch = relu_bwd_f32_launch;
  v.supported = relu_bwd_f32_supported;
  v.query_workspace = relu_bwd_workspace;
  return v;
}

// -------------------------
// Variant: F16 naive
// -------------------------
static bool relu_bwd_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void*) {
  return relu_bwd_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f16_contig);
}

static Status relu_bwd_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void*,
    void*, size_t,
    cudaStream_t stream) {

  if (!relu_bwd_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f16_contig)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& dOut = inputs[0];
  const TensorDesc& Y    = inputs[1];
  TensorDesc& dY         = outputs[0];

  int64_t numel = 0;
  if (!compute_numel(Y, &numel)) return Status::InvalidArgument;

  constexpr int kThreads = 256;
  int64_t blocks64 = (numel + kThreads - 1) / kThreads;
  if (blocks64 < 1) blocks64 = 1;
  if (blocks64 > 65535) blocks64 = 65535;
  const int blocks = (int)blocks64;

  cudaGetLastError(); // clear
  relu_bwd_impl::relu_bwd_f16_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half*)Y.data, (const __half*)dOut.data, (__half*)dY.data, numel);

  return cuda_last_status();
}

KernelVariant make_relu_bwd_f16_variant() {
  KernelVariant v{};
  v.name = "relu_bwd_f16_naive";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = 0;
  v.launch = relu_bwd_f16_launch;
  v.supported = relu_bwd_f16_supported;
  v.query_workspace = relu_bwd_workspace;
  return v;
}

// -------------------------
// Variant: F16 vec2 (half2)
// -------------------------
static bool relu_bwd_f16_vec2_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void*) {
  return relu_bwd_f16_vec2_check(inputs, num_inputs, outputs, num_outputs);
}

static Status relu_bwd_f16_vec2_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void*,
    void*, size_t,
    cudaStream_t stream) {

  if (!relu_bwd_f16_vec2_check(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& dOut = inputs[0];
  const TensorDesc& Y    = inputs[1];
  TensorDesc& dY         = outputs[0];

  int64_t numel = 0;
  if (!compute_numel(Y, &numel)) return Status::InvalidArgument;

  const int64_t numel2 = numel / 2;

  constexpr int kThreads = 256;
  int64_t blocks64 = (numel2 + kThreads - 1) / kThreads;
  if (blocks64 < 1) blocks64 = 1;
  if (blocks64 > 65535) blocks64 = 65535;
  const int blocks = (int)blocks64;

  cudaGetLastError(); // clear
  relu_bwd_impl::relu_bwd_f16x2_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half2*)Y.data, (const __half2*)dOut.data, (__half2*)dY.data, numel2);

  return cuda_last_status();
}

KernelVariant make_relu_bwd_f16_vec2_variant() {
  KernelVariant v{};
  v.name = "relu_bwd_f16_vec2_half2";
  v.priority = 10;
  v.flags = 0;
  v.expected_attr_schema_id = 0;
  v.launch = relu_bwd_f16_vec2_launch;
  v.supported = relu_bwd_f16_vec2_supported;
  v.query_workspace = relu_bwd_workspace;
  return v;
}

} // namespace aicf::cuda
