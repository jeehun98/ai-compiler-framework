// ============================================================================
// src/backends/cuda/ops/mse_grad/launcher.cu  (core-free / AttrBlob ABI)
// - attrs: AttrBlob(schema_id + raw bytes)
// - supports f32 + f16 (half2 fastpath when safe)
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>   // memcpy

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

// ============================================================================
// AttrBlob schema: 'MSEG'
// - schema_id == 0       => default scale = 2.0 / numel
// - schema_id == 'MSEG'  => payload: float32 scale
// Python pack: struct.pack("<f", scale)
// ============================================================================
static constexpr uint32_t kAttrSchema_MseGrad = 0x4745534Du; // 'MSEG'

struct MseGradAttrV0 {
  float scale;
};

static inline bool read_scale_attr(const void* attr, float* out_scale) {
  if (!out_scale) return false;
  const AttrBlob* ab = static_cast<const AttrBlob*>(attr);
  if (!ab) return false;

  if (ab->schema_id != kAttrSchema_MseGrad) return false;
  if (!ab->data || ab->bytes < (uint32_t)sizeof(MseGradAttrV0)) return false;

  MseGradAttrV0 a{};
  std::memcpy(&a, ab->data, sizeof(MseGradAttrV0));
  *out_scale = a.scale;
  return true;
}

// ============================================================================
// kernels
// ============================================================================
namespace mse_grad_impl {

__global__ void mse_grad_f32_kernel(const float* __restrict__ pred,
                                    const float* __restrict__ target,
                                    float* __restrict__ dPred,
                                    int64_t numel,
                                    float scale) {
  for (int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
       i < numel;
       i += (int64_t)blockDim.x * (int64_t)gridDim.x) {
    dPred[i] = (pred[i] - target[i]) * scale;
  }
}

__global__ void mse_grad_f16_kernel(const __half* __restrict__ pred,
                                    const __half* __restrict__ target,
                                    __half* __restrict__ dPred,
                                    int64_t numel,
                                    float scale) {
  for (int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
       i < numel;
       i += (int64_t)blockDim.x * (int64_t)gridDim.x) {
    const float p = __half2float(pred[i]);
    const float t = __half2float(target[i]);
    dPred[i] = __float2half_rn((p - t) * scale);
  }
}

__global__ void mse_grad_f16x2_kernel(const __half2* __restrict__ pred,
                                      const __half2* __restrict__ target,
                                      __half2* __restrict__ dPred,
                                      int64_t numel2,
                                      float scale) {
  const __half2 s2 = __float2half2_rn(scale);
  for (int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
       i < numel2;
       i += (int64_t)blockDim.x * (int64_t)gridDim.x) {
    dPred[i] = __hmul2(__hsub2(pred[i], target[i]), s2);
  }
}

} // namespace mse_grad_impl

// ============================================================================
// helpers
// ============================================================================
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

static inline bool mse_grad_check_dt(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    bool (*is_ok)(const TensorDesc&)) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& P = inputs[0];
  const TensorDesc& T = inputs[1];
  const TensorDesc& G = outputs[0];

  if (!is_ok(P) || !is_ok(T) || !is_ok(G)) return false;
  if (P.rank() < 1) return false;
  if (!same_shape(P, T)) return false;
  if (!same_shape(P, G)) return false;

  int64_t numel = 0;
  if (!compute_numel(P, &numel)) return false;
  return true;
}

static inline bool mse_grad_vec2_ok(const TensorDesc& P, const TensorDesc& T, const TensorDesc& G) {
  int64_t numel = 0;
  if (!compute_numel(P, &numel)) return false;
  if ((numel & 1) != 0) return false;

  constexpr size_t kAlign = 4;
  if (!is_aligned_ptr(P.data, kAlign)) return false;
  if (!is_aligned_ptr(T.data, kAlign)) return false;
  if (!is_aligned_ptr(G.data, kAlign)) return false;
  return true;
}

static size_t mse_grad_workspace(const TensorDesc*, int, const void*) { return 0; }

// blocks helper (same as 너 코드)
static inline int compute_blocks_1d(int64_t n, int threads) {
  int64_t blocks64 = (n + threads - 1) / threads;
  if (blocks64 < 1) blocks64 = 1;
  const int64_t kMaxBlocks = 4096;
  if (blocks64 > kMaxBlocks) blocks64 = kMaxBlocks;
  return (int)blocks64;
}

// ============================================================================
// Variant: F32
// ============================================================================
static bool mse_grad_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return mse_grad_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f32_contig);
}

static Status mse_grad_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!mse_grad_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f32_contig)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& P = inputs[0];
  const TensorDesc& T = inputs[1];
  TensorDesc& G = outputs[0];

  int64_t numel = 0;
  if (!compute_numel(P, &numel)) return Status::InvalidArgument;

  float scale = 0.0f;
  const bool has_scale = read_scale_attr(attr, &scale);
  if (!has_scale) scale = 2.0f / (float)numel;

  constexpr int kThreads = 256;
  const int blocks = compute_blocks_1d(numel, kThreads);

  cudaGetLastError(); // clear
  mse_grad_impl::mse_grad_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)P.data, (const float*)T.data, (float*)G.data, numel, scale);

  return cuda_last_status();
}

KernelVariant make_mse_grad_f32_variant() {
  KernelVariant v{};
  v.name = "mse_grad_f32";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = 0; // accept schema 0 (default) or MSEG (we parse)
  v.launch = mse_grad_f32_launch;
  v.supported = mse_grad_f32_supported;
  v.query_workspace = mse_grad_workspace;
  return v;
}

// ============================================================================
// Variant: F16
// ============================================================================
static bool mse_grad_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return mse_grad_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f16_contig);
}

static Status mse_grad_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!mse_grad_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f16_contig)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& P = inputs[0];
  const TensorDesc& T = inputs[1];
  TensorDesc& G = outputs[0];

  int64_t numel = 0;
  if (!compute_numel(P, &numel)) return Status::InvalidArgument;

  float scale = 0.0f;
  const bool has_scale = read_scale_attr(attr, &scale);
  if (!has_scale) scale = 2.0f / (float)numel;

  constexpr int kThreads = 256;

  cudaGetLastError(); // clear
  if (mse_grad_vec2_ok(P, T, G)) {
    const int64_t numel2 = numel / 2;
    const int blocks = compute_blocks_1d(numel2, kThreads);
    mse_grad_impl::mse_grad_f16x2_kernel<<<blocks, kThreads, 0, stream>>>(
        (const __half2*)P.data, (const __half2*)T.data, (__half2*)G.data, numel2, scale);
  } else {
    const int blocks = compute_blocks_1d(numel, kThreads);
    mse_grad_impl::mse_grad_f16_kernel<<<blocks, kThreads, 0, stream>>>(
        (const __half*)P.data, (const __half*)T.data, (__half*)G.data, numel, scale);
  }

  return cuda_last_status();
}

KernelVariant make_mse_grad_f16_variant() {
  KernelVariant v{};
  v.name = "mse_grad_f16";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = 0;
  v.launch = mse_grad_f16_launch;
  v.supported = mse_grad_f16_supported;
  v.query_workspace = mse_grad_workspace;
  return v;
}

} // namespace aicf::cuda
