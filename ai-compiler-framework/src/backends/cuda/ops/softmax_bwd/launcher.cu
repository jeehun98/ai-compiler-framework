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

static inline Status cuda_to_status(cudaError_t e) {
  return (e == cudaSuccess) ? Status::Ok : Status::Internal;
}
static inline Status cuda_last_status() {
  return cuda_to_status(cudaGetLastError());
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

static inline int64_t last_dim(const TensorDesc& T) {
  const int64_t r = T.rank();
  return (r >= 1) ? T.shape[r - 1] : 0;
}

static size_t softmax_bwd_workspace(const TensorDesc*, int, const void*) { return 0; }

// ============================================================================
// kernels
// ============================================================================
namespace softmax_bwd_impl {

static __forceinline__ __device__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

static __forceinline__ __device__ float block_reduce_sum(float v) {
  __shared__ float smem[32];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;

  v = warp_reduce_sum(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();

  float out = 0.0f;
  if (warp == 0) {
    int warps = (blockDim.x + 31) >> 5;
    out = (lane < warps) ? smem[lane] : 0.0f;
    out = warp_reduce_sum(out);
  }
  if (threadIdx.x == 0) smem[0] = out;
  __syncthreads();
  return smem[0];
}

__global__ void softmax_bwd_lastdim_f32_kernel(const float* __restrict__ y,
                                              const float* __restrict__ dy,
                                              float* __restrict__ dx,
                                              int64_t rows,
                                              int64_t cols) {
  const int64_t row = (int64_t)blockIdx.x;
  if (row >= rows) return;

  const int64_t base = row * cols;

  // dot = sum(dy * y)
  float t = 0.0f;
  for (int64_t c = (int64_t)threadIdx.x; c < cols; c += (int64_t)blockDim.x) {
    t += dy[base + c] * y[base + c];
  }
  const float dot = block_reduce_sum(t);

  // dx = y * (dy - dot)
  for (int64_t c = (int64_t)threadIdx.x; c < cols; c += (int64_t)blockDim.x) {
    float yi = y[base + c];
    float dyi = dy[base + c];
    dx[base + c] = yi * (dyi - dot);
  }
}

__global__ void softmax_bwd_lastdim_f16_kernel(const __half* __restrict__ y,
                                              const __half* __restrict__ dy,
                                              __half* __restrict__ dx,
                                              int64_t rows,
                                              int64_t cols) {
  const int64_t row = (int64_t)blockIdx.x;
  if (row >= rows) return;

  const int64_t base = row * cols;

  float t = 0.0f;
  for (int64_t c = (int64_t)threadIdx.x; c < cols; c += (int64_t)blockDim.x) {
    float yi = __half2float(y[base + c]);
    float dyi = __half2float(dy[base + c]);
    t += dyi * yi;
  }
  const float dot = block_reduce_sum(t);

  for (int64_t c = (int64_t)threadIdx.x; c < cols; c += (int64_t)blockDim.x) {
    float yi = __half2float(y[base + c]);
    float dyi = __half2float(dy[base + c]);
    float out = yi * (dyi - dot);
    dx[base + c] = __float2half_rn(out);
  }
}

} // namespace softmax_bwd_impl

// ============================================================================
// Variant checks
// Contract: inputs=(Y, dY), outputs=(dX)
// - all same shape, contiguous
// - forbid alias: dX must not alias Y or dY (간단히 안전하게)
// ============================================================================
static inline bool softmax_bwd_check_f32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& Y  = inputs[0];
  const TensorDesc& dY = inputs[1];
  const TensorDesc& dX = outputs[0];

  if (!is_f32_contig_anyrank(Y) || !is_f32_contig_anyrank(dY) || !is_f32_contig_anyrank(dX)) return false;
  if (!same_shape(Y, dY) || !same_shape(Y, dX)) return false;

  int64_t numel = 0;
  if (!compute_numel(Y, &numel)) return false;
  if (numel <= 0) return false;

  const int64_t cols = last_dim(Y);
  if (cols <= 0) return false;
  if ((numel % cols) != 0) return false;

  return true;
}

static bool softmax_bwd_supported_f32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return softmax_bwd_check_f32(inputs, num_inputs, outputs, num_outputs);
}

static Status softmax_bwd_launch_f32(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void*, size_t,
    cudaStream_t stream) {

  if (!softmax_bwd_check_f32(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& Y  = inputs[0];
  const TensorDesc& dY = inputs[1];
  TensorDesc& dX = outputs[0];

  // forbid alias (단순 정책)
  if (dX.data == Y.data)  return Status::InvalidArgument;
  if (dX.data == dY.data) return Status::InvalidArgument;

  int64_t numel = 0;
  (void)compute_numel(Y, &numel);
  const int64_t cols = last_dim(Y);
  const int64_t rows = numel / cols;

  constexpr int kThreads = 256;
  int blocks = (rows > 0) ? (int)rows : 1;
  if (blocks > 2147483647) blocks = 2147483647;

  cudaGetLastError(); // clear
  softmax_bwd_impl::softmax_bwd_lastdim_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)Y.data, (const float*)dY.data, (float*)dX.data, rows, cols);

  return cuda_last_status();
}

KernelVariant make_softmax_bwd_f32_variant() {
  KernelVariant v{};
  v.name = "softmax_bwd_lastdim_f32";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = 0;
  v.launch = softmax_bwd_launch_f32;
  v.supported = softmax_bwd_supported_f32;
  v.query_workspace = softmax_bwd_workspace;
  return v;
}

// ---------------- f16 ----------------
static inline bool softmax_bwd_check_f16(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& Y  = inputs[0];
  const TensorDesc& dY = inputs[1];
  const TensorDesc& dX = outputs[0];

  if (!is_f16_contig_anyrank(Y) || !is_f16_contig_anyrank(dY) || !is_f16_contig_anyrank(dX)) return false;
  if (!same_shape(Y, dY) || !same_shape(Y, dX)) return false;

  int64_t numel = 0;
  if (!compute_numel(Y, &numel)) return false;
  if (numel <= 0) return false;

  const int64_t cols = last_dim(Y);
  if (cols <= 0) return false;
  if ((numel % cols) != 0) return false;

  return true;
}

static bool softmax_bwd_supported_f16(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return softmax_bwd_check_f16(inputs, num_inputs, outputs, num_outputs);
}

static Status softmax_bwd_launch_f16(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void*, size_t,
    cudaStream_t stream) {

  if (!softmax_bwd_check_f16(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& Y  = inputs[0];
  const TensorDesc& dY = inputs[1];
  TensorDesc& dX = outputs[0];

  if (dX.data == Y.data)  return Status::InvalidArgument;
  if (dX.data == dY.data) return Status::InvalidArgument;

  int64_t numel = 0;
  (void)compute_numel(Y, &numel);
  const int64_t cols = last_dim(Y);
  const int64_t rows = numel / cols;

  constexpr int kThreads = 256;
  int blocks = (rows > 0) ? (int)rows : 1;
  if (blocks > 2147483647) blocks = 2147483647;

  cudaGetLastError(); // clear
  softmax_bwd_impl::softmax_bwd_lastdim_f16_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half*)Y.data, (const __half*)dY.data, (__half*)dX.data, rows, cols);

  return cuda_last_status();
}

KernelVariant make_softmax_bwd_f16_variant() {
  KernelVariant v{};
  v.name = "softmax_bwd_lastdim_f16";
  v.priority = 10;
  v.flags = 0;
  v.expected_attr_schema_id = 0;
  v.launch = softmax_bwd_launch_f16;
  v.supported = softmax_bwd_supported_f16;
  v.query_workspace = softmax_bwd_workspace;
  return v;
}

} // namespace aicf::cuda
