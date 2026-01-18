#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>

#include <aicf/backends/cuda/registry/status.hpp>
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_blob.hpp>

#include "kernels.cuh"

namespace aicf::cuda {

// -------------------------
// cuda error -> Status (core-free)
// -------------------------
static inline Status cuda_to_status(cudaError_t e) {
  return (e == cudaSuccess) ? Status::Ok : Status::Internal;
}
static inline Status cuda_last_status() {
  return cuda_to_status(cudaGetLastError());
}

// -------------------------
// AttrBlob schema: SGDStep
// payload: float32 lr
// schema_id: 'SGDS' (0x53444753)  (little-endian pack in python)
// schema_id==0 allowed -> default lr
// -------------------------
static constexpr uint32_t kSchema_SGDS = 0x53444753u;

static inline float read_f32_le(const uint8_t* p) {
  float v;
  std::memcpy(&v, p, sizeof(float));
  return v;
}

static inline float get_lr_from_attr_blob(const AttrBlob& a, float default_lr) {
  if (a.schema_id == 0) return default_lr;
  if (a.schema_id != kSchema_SGDS) return default_lr;
  if (a.bytes < 4 || !a.data) return default_lr;
  return read_f32_le(static_cast<const uint8_t*>(a.data));
}

static inline AttrBlob as_attr_blob(const void* attr) {
  if (!attr) return AttrBlob{0, 0, nullptr};
  return *static_cast<const AttrBlob*>(attr);
}

// -------------------------
// helpers
// -------------------------
static inline bool same_shape(const TensorDesc& A, const TensorDesc& B) {
  if (A.rank() != B.rank()) return false;
  for (int64_t i = 0; i < A.rank(); ++i) {
    if (A.shape[i] != B.shape[i]) return false;
  }
  return true;
}

static inline bool compute_numel(const TensorDesc& T, int64_t* out) {
  if (!out) return false;
  const int64_t r = T.rank();
  if (r <= 0) return false;
  int64_t n = 1;
  for (int64_t i = 0; i < r; ++i) {
    const int64_t d = T.shape[i];
    if (d <= 0) return false;
    n *= d;
  }
  *out = n;
  return true;
}

static inline bool is_f32_contig_anyrank(const TensorDesc& T) {
  return (T.dtype == DType::kF32) && T.contiguous && (T.rank() >= 1);
}
static inline bool is_f16_contig_anyrank(const TensorDesc& T) {
  return (T.dtype == DType::kF16) && T.contiguous && (T.rank() >= 1);
}

static inline bool ptr_aligned_4(const void* p) {
  return ((uintptr_t)p & 0x3u) == 0;
}

// grid sizing (grid-stride kernel)
static inline int choose_blocks_1d(int64_t numel, int threads) {
  int64_t blocks64 = (numel + threads - 1) / threads;
  if (blocks64 < 1) blocks64 = 1;
  const int64_t kMaxBlocks = 65535;
  if (blocks64 > kMaxBlocks) blocks64 = kMaxBlocks;
  return (int)blocks64;
}

static size_t sgd_step_workspace(const TensorDesc*, int, const void*) { return 0; }

// ============================================================================
// kernels (definitions live here)
// ============================================================================
namespace sgd_step_impl {

static __forceinline__ __device__ int64_t global_tid_1d() {
  return (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
}
static __forceinline__ __device__ int64_t global_stride_1d() {
  return (int64_t)gridDim.x * (int64_t)blockDim.x;
}

__global__ void sgd_step_f32_kernel(const float* __restrict__ param_in,
                                   const float* __restrict__ grad,
                                   float* __restrict__ param_out,
                                   int64_t numel,
                                   float lr) {
  for (int64_t i = global_tid_1d(); i < numel; i += global_stride_1d()) {
    param_out[i] = param_in[i] - lr * grad[i];
  }
}

__global__ void sgd_step_f16_kernel(const __half* __restrict__ param_in,
                                   const __half* __restrict__ grad,
                                   __half* __restrict__ param_out,
                                   int64_t numel,
                                   float lr) {
  for (int64_t i = global_tid_1d(); i < numel; i += global_stride_1d()) {
    float p = __half2float(param_in[i]);
    float g = __half2float(grad[i]);
    p -= lr * g;
    param_out[i] = __float2half_rn(p);
  }
}

__global__ void sgd_step_f16_half2_kernel(const __half2* __restrict__ param_in2,
                                         const __half2* __restrict__ grad2,
                                         __half2* __restrict__ param_out2,
                                         int64_t numel2,
                                         float lr) {
  for (int64_t i = global_tid_1d(); i < numel2; i += global_stride_1d()) {
    __half2 p2 = param_in2[i];
    __half2 g2 = grad2[i];

    float2 pf = __half22float2(p2);
    float2 gf = __half22float2(g2);

    pf.x -= lr * gf.x;
    pf.y -= lr * gf.y;

    param_out2[i] = __floats2half2_rn(pf.x, pf.y);
  }
}

} // namespace sgd_step_impl

// ============================================================================
// Variant: SGDStep f32
// Contract: inputs=(P,G), outputs=(O)
//   O = P - lr * G
// Allows in-place: O may alias P.
// Forbids: O aliases G.
// ============================================================================
static inline bool sgd_step_check_f32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& P = inputs[0];
  const TensorDesc& G = inputs[1];
  const TensorDesc& O = outputs[0];

  if (!is_f32_contig_anyrank(P) || !is_f32_contig_anyrank(G) || !is_f32_contig_anyrank(O)) return false;
  if (!same_shape(P, G)) return false;
  if (!same_shape(P, O)) return false;

  int64_t numel = 0;
  if (!compute_numel(P, &numel)) return false;
  return (numel > 0);
}

static bool sgd_step_supported_f32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return sgd_step_check_f32(inputs, num_inputs, outputs, num_outputs);
}

static Status sgd_step_launch_f32(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!sgd_step_check_f32(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& P = inputs[0];
  const TensorDesc& G = inputs[1];
  TensorDesc& O = outputs[0];

  if (O.data == G.data) return Status::InvalidArgument;

  int64_t numel = 0;
  (void)compute_numel(P, &numel);

  const AttrBlob a = as_attr_blob(attr);
  const float lr = get_lr_from_attr_blob(a, 1e-3f);

  constexpr int kThreads = 256;
  const int blocks = choose_blocks_1d(numel, kThreads);

  cudaGetLastError(); // clear
  sgd_step_impl::sgd_step_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)P.data, (const float*)G.data, (float*)O.data, numel, lr);

  return cuda_last_status();
}

KernelVariant make_sgd_step_f32_variant() {
  KernelVariant v{};
  v.name = "sgd_step_f32";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = 0; // allow schema_id==0 (default lr)
  v.launch = sgd_step_launch_f32;
  v.supported = sgd_step_supported_f32;
  v.query_workspace = sgd_step_workspace;
  return v;
}

// ============================================================================
// Variant: SGDStep f16 scalar
// ============================================================================
static inline bool sgd_step_check_f16(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& P = inputs[0];
  const TensorDesc& G = inputs[1];
  const TensorDesc& O = outputs[0];

  if (!is_f16_contig_anyrank(P) || !is_f16_contig_anyrank(G) || !is_f16_contig_anyrank(O)) return false;
  if (!same_shape(P, G)) return false;
  if (!same_shape(P, O)) return false;

  int64_t numel = 0;
  if (!compute_numel(P, &numel)) return false;
  return (numel > 0);
}

static bool sgd_step_supported_f16(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return sgd_step_check_f16(inputs, num_inputs, outputs, num_outputs);
}

static Status sgd_step_launch_f16(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!sgd_step_check_f16(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& P = inputs[0];
  const TensorDesc& G = inputs[1];
  TensorDesc& O = outputs[0];

  if (O.data == G.data) return Status::InvalidArgument;

  int64_t numel = 0;
  (void)compute_numel(P, &numel);

  const AttrBlob a = as_attr_blob(attr);
  const float lr = get_lr_from_attr_blob(a, 1e-3f);

  constexpr int kThreads = 256;
  const int blocks = choose_blocks_1d(numel, kThreads);

  cudaGetLastError(); // clear
  sgd_step_impl::sgd_step_f16_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half*)P.data, (const __half*)G.data, (__half*)O.data, numel, lr);

  return cuda_last_status();
}

KernelVariant make_sgd_step_f16_variant() {
  KernelVariant v{};
  v.name = "sgd_step_f16";
  v.priority = 10;
  v.flags = 0;
  v.expected_attr_schema_id = 0; // allow schema_id==0
  v.launch = sgd_step_launch_f16;
  v.supported = sgd_step_supported_f16;
  v.query_workspace = sgd_step_workspace;
  return v;
}

// ============================================================================
// Variant: SGDStep f16 half2
// Eligibility: numel even + 4B aligned pointers.
// ============================================================================
static inline bool sgd_step_check_f16_half2(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!sgd_step_check_f16(inputs, num_inputs, outputs, num_outputs)) return false;

  const TensorDesc& P = inputs[0];
  const TensorDesc& G = inputs[1];
  const TensorDesc& O = outputs[0];

  int64_t numel = 0;
  (void)compute_numel(P, &numel);

  if ((numel & 1) != 0) return false;
  if (!ptr_aligned_4(P.data)) return false;
  if (!ptr_aligned_4(G.data)) return false;
  if (!ptr_aligned_4(O.data)) return false;

  return true;
}

static bool sgd_step_supported_f16_half2(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return sgd_step_check_f16_half2(inputs, num_inputs, outputs, num_outputs);
}

static Status sgd_step_launch_f16_half2(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!sgd_step_check_f16_half2(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& P = inputs[0];
  const TensorDesc& G = inputs[1];
  TensorDesc& O = outputs[0];

  if (O.data == G.data) return Status::InvalidArgument;

  int64_t numel = 0;
  (void)compute_numel(P, &numel);
  const int64_t numel2 = numel >> 1;

  const AttrBlob a = as_attr_blob(attr);
  const float lr = get_lr_from_attr_blob(a, 1e-3f);

  constexpr int kThreads = 256;
  const int blocks = choose_blocks_1d(numel2, kThreads);

  cudaGetLastError(); // clear
  sgd_step_impl::sgd_step_f16_half2_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half2*)P.data, (const __half2*)G.data, (__half2*)O.data, numel2, lr);

  return cuda_last_status();
}

KernelVariant make_sgd_step_f16_half2_variant() {
  KernelVariant v{};
  v.name = "sgd_step_f16_half2";
  v.priority = 20;
  v.flags = 0;
  v.expected_attr_schema_id = 0; // allow schema_id==0
  v.launch = sgd_step_launch_f16_half2;
  v.supported = sgd_step_supported_f16_half2;
  v.query_workspace = sgd_step_workspace;
  return v;
}

} // namespace aicf::cuda
