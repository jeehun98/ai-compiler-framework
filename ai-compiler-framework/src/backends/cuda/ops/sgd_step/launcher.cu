#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include <aicf/core/status.hpp>

// registry glue
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_pack.hpp>

// common shim
#include "aicf/backends/cuda/ops/_common/shim/status.hpp"

#include "kernels.cuh"

#include <string_view>

namespace aicf::cuda {

// -------------------------
// kernels (definitions live here)
// -------------------------
namespace sgd_step_impl {

static __forceinline__ __device__ int64_t global_tid_1d() {
  return (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
}
static __forceinline__ __device__ int64_t global_stride_1d() {
  return (int64_t)gridDim.x * (int64_t)blockDim.x;
}

// out-of-place: out = in - lr * grad
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

// half2 out-of-place: out2 = in2 - lr * grad2 (elementwise)
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

// -------------------------
// Attr helper: lr (float)
// -------------------------
static inline float attr_get_lr_f32(const void* attr, float default_lr) {
  if (!attr) return default_lr;
  const auto* pack = static_cast<const aicf::cuda::AttrPack*>(attr);
  float lr = default_lr;
  (void)pack->get_f32("lr", &lr);
  return lr;
}

// -------------------------
// Shape helpers
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

// grid sizing: keep blocks reasonable but don't truncate work (grid-stride handles it)
static inline int choose_blocks_1d(int64_t numel, int threads) {
  int64_t blocks64 = (numel + threads - 1) / threads;
  // cap blocks to something sane for launch overhead (grid-stride covers full range)
  const int kMaxBlocks = 65535;
  int blocks = (blocks64 > (int64_t)kMaxBlocks) ? kMaxBlocks : (int)blocks64;
  if (blocks < 1) blocks = 1;
  return blocks;
}

static size_t sgd_step_workspace(const TensorDesc*, int, const void*) { return 0; }

// ============================================================================
// Variant: SGDStep f32
// Contract: inputs=(P,G), outputs=(O)
//   O = P - lr * G
// Allows in-place: O may alias P.
// Forbids: O aliases G (would clobber grad while reading).
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

static aicf::Status sgd_step_launch_f32(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!sgd_step_check_f32(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& P = inputs[0];
  const TensorDesc& G = inputs[1];
  TensorDesc& O = outputs[0];

  // forbid alias grad==out (unsafe)
  if (O.data == G.data) return aicf::Status::InvalidArgument;

  int64_t numel = 0;
  (void)compute_numel(P, &numel);

  const float lr = attr_get_lr_f32(attr, 1e-3f);

  constexpr int kThreads = 256;
  const int blocks = choose_blocks_1d(numel, kThreads);

  sgd_step_impl::sgd_step_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)P.data, (const float*)G.data, (float*)O.data, numel, lr);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_sgd_step_f32_variant() {
  KernelVariant v{};
  v.name = "sgd_step_f32";
  v.priority = 0;
  v.flags = 0;
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

static aicf::Status sgd_step_launch_f16(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!sgd_step_check_f16(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& P = inputs[0];
  const TensorDesc& G = inputs[1];
  TensorDesc& O = outputs[0];

  if (O.data == G.data) return aicf::Status::InvalidArgument;

  int64_t numel = 0;
  (void)compute_numel(P, &numel);

  const float lr = attr_get_lr_f32(attr, 1e-3f);

  constexpr int kThreads = 256;
  const int blocks = choose_blocks_1d(numel, kThreads);

  sgd_step_impl::sgd_step_f16_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half*)P.data, (const __half*)G.data, (__half*)O.data, numel, lr);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_sgd_step_f16_variant() {
  KernelVariant v{};
  v.name = "sgd_step_f16";
  v.priority = 10; // scalar f16
  v.flags = 0;
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

static aicf::Status sgd_step_launch_f16_half2(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!sgd_step_check_f16_half2(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& P = inputs[0];
  const TensorDesc& G = inputs[1];
  TensorDesc& O = outputs[0];

  if (O.data == G.data) return aicf::Status::InvalidArgument;

  int64_t numel = 0;
  (void)compute_numel(P, &numel);
  const int64_t numel2 = numel >> 1;

  const float lr = attr_get_lr_f32(attr, 1e-3f);

  constexpr int kThreads = 256;
  const int blocks = choose_blocks_1d(numel2, kThreads);

  sgd_step_impl::sgd_step_f16_half2_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half2*)P.data, (const __half2*)G.data, (__half2*)O.data, numel2, lr);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_sgd_step_f16_half2_variant() {
  KernelVariant v{};
  v.name = "sgd_step_f16_half2";
  v.priority = 20; // prefer half2 over scalar f16
  v.flags = 0;
  v.launch = sgd_step_launch_f16_half2;
  v.supported = sgd_step_supported_f16_half2;
  v.query_workspace = sgd_step_workspace;
  return v;
}

} // namespace aicf::cuda
