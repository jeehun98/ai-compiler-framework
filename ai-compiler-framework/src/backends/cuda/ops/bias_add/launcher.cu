// ============================================================================
// src/backends/cuda/ops/bias_add/launcher.cu  (core-free / AttrBlob ABI)
// - attrs: AttrBlob(schema_id + raw bytes)
//   * schema_id == 0 => default axis=-1 (last dim)
//   * schema_id == 'BADD' => payload: int64 axis
// - supports:
//   * f32 contiguous, rank>=2
//   * f16 contiguous, rank>=2
//   * f16 half2 fastpath (last dim even + 4B aligned pointers)
// - axis: only last-dim allowed (-1 or rank-1)
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>   // memcpy
#include <cinttypes> // uintptr_t

#include <aicf/backends/cuda/ops/bias_add/api.hpp> // keep if you expose this op API

#include <aicf/backends/cuda/registry/status.hpp>
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_blob.hpp>

#include "kernels.cuh"

namespace aicf::cuda {

// -------------------------
// kernels (definitions live here)
// -------------------------
namespace bias_add_impl {

__global__ void bias_add_f32_kernel(const float* __restrict__ Y,
                                   const float* __restrict__ bias,
                                   float* __restrict__ Out,
                                   int M, int N) {
  const int64_t total = (int64_t)M * (int64_t)N;
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t step = (int64_t)blockDim.x * gridDim.x;

  for (; tid < total; tid += step) {
    const int col = (int)(tid % N);
    Out[tid] = Y[tid] + bias[col];
  }
}

__global__ void bias_add_f16_kernel(const __half* __restrict__ Y,
                                   const __half* __restrict__ bias,
                                   __half* __restrict__ Out,
                                   int M, int N) {
  const int64_t total = (int64_t)M * (int64_t)N;
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t step = (int64_t)blockDim.x * gridDim.x;

  for (; tid < total; tid += step) {
    const int col = (int)(tid % N);
    Out[tid] = __hadd(Y[tid], bias[col]);
  }
}

__global__ void bias_add_f16x2_kernel(const __half2* __restrict__ Y,
                                      const __half2* __restrict__ bias,
                                      __half2* __restrict__ Out,
                                      int M, int N2) {
  const int64_t total2 = (int64_t)M * (int64_t)N2;
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t step = (int64_t)blockDim.x * gridDim.x;

  for (; tid < total2; tid += step) {
    const int col2 = (int)(tid % N2);
    Out[tid] = __hadd2(Y[tid], bias[col2]);
  }
}

} // namespace bias_add_impl


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

// ============================================================================
// AttrBlob schema: 'BADD'
// - Python packs: struct.pack("<q", axis)  (int64)
// ============================================================================
static constexpr uint32_t kAttrSchema_BiasAdd = 0x44444142u; // 'BADD'

struct BiasAddAttrV0 {
  int64_t axis;
};

static inline int64_t read_axis_default_lastdim(const void* attr) {
  // default: last dim (-1)
  int64_t axis = -1;

  const AttrBlob* ab = static_cast<const AttrBlob*>(attr);
  if (!ab) return axis;

  // accept schema_id==0 as "no attrs"
  if (ab->schema_id != 0 && ab->schema_id != kAttrSchema_BiasAdd) {
    return axis;
  }

  if (!ab->data || ab->bytes < (uint32_t)sizeof(BiasAddAttrV0)) {
    return axis;
  }

  BiasAddAttrV0 a{};
  std::memcpy(&a, ab->data, sizeof(BiasAddAttrV0));
  return a.axis;
}

// ============================================================================
// last-dim-only helpers (safe)
// ============================================================================
static inline bool is_contig_rank_ge2(const TensorDesc& T, DType dt) {
  return (T.dtype == dt) && T.contiguous && (T.rank() >= 2);
}

static inline bool axis_is_last_dim_only(const TensorDesc& Y, int64_t axis_raw) {
  const int64_t r = Y.rank();
  if (r < 2) return false;
  const int64_t last = r - 1;
  return (axis_raw == -1) || (axis_raw == last);
}

static inline bool compute_MN_last_dim(const TensorDesc& Y, int64_t* out_M, int64_t* out_N) {
  const int64_t r = Y.rank();
  if (r < 2) return false;

  const int64_t N = Y.shape[r - 1];
  if (N <= 0) return false;

  int64_t M = 1;
  for (int64_t i = 0; i < r - 1; ++i) {
    const int64_t d = Y.shape[i];
    if (d <= 0) return false;
    M *= d;
  }
  if (M <= 0) return false;

  *out_M = M;
  *out_N = N;
  return true;
}

static inline bool bias_add_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    int64_t axis_raw,
    DType dt) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& Y = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& O = outputs[0];

  if (!is_contig_rank_ge2(Y, dt)) return false;
  if (!is_contig_rank_ge2(O, dt)) return false;

  // bias: same dtype, contig 1D
  if (!(B.contiguous && B.rank() == 1 && B.dtype == dt)) return false;

  if (!axis_is_last_dim_only(Y, axis_raw)) return false;

  // output must match Y shape exactly
  if (O.rank() != Y.rank()) return false;
  for (int64_t i = 0; i < Y.rank(); ++i) {
    if (O.shape[i] != Y.shape[i]) return false;
  }

  int64_t M = 0, N = 0;
  if (!compute_MN_last_dim(Y, &M, &N)) return false;

  if (B.shape[0] != N) return false;
  return true;
}

static size_t bias_add_workspace(const TensorDesc*, int, const void*) { return 0; }

// ============================================================================
// Variant: F32
// ============================================================================
static bool bias_add_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  const int64_t axis = read_axis_default_lastdim(attr);
  return bias_add_check(inputs, num_inputs, outputs, num_outputs, axis, DType::kF32);
}

static Status bias_add_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  const int64_t axis = read_axis_default_lastdim(attr);

  if (!bias_add_check(inputs, num_inputs, outputs, num_outputs, axis, DType::kF32)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& Y = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  int64_t M64 = 0, N64 = 0;
  if (!compute_MN_last_dim(Y, &M64, &N64)) return Status::InvalidArgument;

  const int M = (int)M64;
  const int N = (int)N64;

  constexpr int kThreads = 256;
  const int64_t total = (int64_t)M * (int64_t)N;
  int blocks = (int)((total + kThreads - 1) / kThreads);
  if (blocks > 65535) blocks = 65535;

  cudaGetLastError(); // clear
  bias_add_impl::bias_add_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)Y.data, (const float*)B.data, (float*)O.data, M, N);

  return cuda_last_status();
}

KernelVariant make_bias_add_f32_variant() {
  KernelVariant v{};
  v.name = "bias_add_f32";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = 0; // accept schema 0 or BADD (we parse both)
  v.launch = bias_add_f32_launch;
  v.supported = bias_add_f32_supported;
  v.query_workspace = bias_add_workspace;
  return v;
}

// ============================================================================
// Variant: F16 naive
// ============================================================================
static bool bias_add_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  const int64_t axis = read_axis_default_lastdim(attr);
  return bias_add_check(inputs, num_inputs, outputs, num_outputs, axis, DType::kF16);
}

static Status bias_add_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  const int64_t axis = read_axis_default_lastdim(attr);

  if (!bias_add_check(inputs, num_inputs, outputs, num_outputs, axis, DType::kF16)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& Y = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  int64_t M64 = 0, N64 = 0;
  if (!compute_MN_last_dim(Y, &M64, &N64)) return Status::InvalidArgument;

  const int M = (int)M64;
  const int N = (int)N64;

  constexpr int kThreads = 256;
  const int64_t total = (int64_t)M * (int64_t)N;
  int blocks = (int)((total + kThreads - 1) / kThreads);
  if (blocks > 65535) blocks = 65535;

  cudaGetLastError(); // clear
  bias_add_impl::bias_add_f16_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half*)Y.data, (const __half*)B.data, (__half*)O.data, M, N);

  return cuda_last_status();
}

KernelVariant make_bias_add_f16_variant() {
  KernelVariant v{};
  v.name = "bias_add_f16_naive";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = 0;
  v.launch = bias_add_f16_launch;
  v.supported = bias_add_f16_supported;
  v.query_workspace = bias_add_workspace;
  return v;
}

// ============================================================================
// Variant: F16 half2 (vec2)
// ============================================================================
static inline bool bias_add_f16_vec2_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    int64_t axis_raw) {

  if (!bias_add_check(inputs, num_inputs, outputs, num_outputs, axis_raw, DType::kF16)) {
    return false;
  }

  const TensorDesc& Y = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& O = outputs[0];

  // last dim even
  const int64_t r = Y.rank();
  const int64_t N = Y.shape[r - 1];
  if ((N & 1) != 0) return false;

  // 4B align
  constexpr size_t kAlign = 4;
  if (!is_aligned_ptr(Y.data, kAlign)) return false;
  if (!is_aligned_ptr(B.data, kAlign)) return false;
  if (!is_aligned_ptr(O.data, kAlign)) return false;

  return true;
}

static bool bias_add_f16_vec2_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  const int64_t axis = read_axis_default_lastdim(attr);
  return bias_add_f16_vec2_check(inputs, num_inputs, outputs, num_outputs, axis);
}

static Status bias_add_f16_vec2_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  const int64_t axis = read_axis_default_lastdim(attr);

  if (!bias_add_f16_vec2_check(inputs, num_inputs, outputs, num_outputs, axis)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& Y = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  int64_t M64 = 0, N64 = 0;
  if (!compute_MN_last_dim(Y, &M64, &N64)) return Status::InvalidArgument;

  const int M = (int)M64;
  const int N = (int)N64;
  const int N2 = N / 2;

  constexpr int kThreads = 256;
  const int64_t total2 = (int64_t)M * (int64_t)N2;
  int blocks = (int)((total2 + kThreads - 1) / kThreads);
  if (blocks > 65535) blocks = 65535;

  cudaGetLastError(); // clear
  bias_add_impl::bias_add_f16x2_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half2*)Y.data, (const __half2*)B.data, (__half2*)O.data, M, N2);

  return cuda_last_status();
}

KernelVariant make_bias_add_f16_vec2_variant() {
  KernelVariant v{};
  v.name = "bias_add_f16_vec2_half2";
  v.priority = 10;
  v.flags = 0;
  v.expected_attr_schema_id = 0;
  v.launch = bias_add_f16_vec2_launch;
  v.supported = bias_add_f16_vec2_supported;
  v.query_workspace = bias_add_workspace;
  return v;
}

} // namespace aicf::cuda
