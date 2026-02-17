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

// workspace: none
static size_t softmax_workspace(const TensorDesc*, int, const void*) { return 0; }

// ============================================================================
// kernels (definitions live here)
// ============================================================================

namespace softmax_impl {

static __forceinline__ __device__ float warp_reduce_max(float v) {
  // full mask
  for (int offset = 16; offset > 0; offset >>= 1) {
    float oth = __shfl_down_sync(0xffffffffu, v, offset);
    v = (v > oth) ? v : oth;
  }
  return v;
}

static __forceinline__ __device__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

static __forceinline__ __device__ float block_reduce_max(float v) {
  // one float per warp
  __shared__ float smem[32]; // up to 1024 threads
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;

  v = warp_reduce_max(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();

  // first warp reduces warp results
  float out = -INFINITY;
  if (warp == 0) {
    int warps = (blockDim.x + 31) >> 5;
    out = (lane < warps) ? smem[lane] : -INFINITY;
    out = warp_reduce_max(out);
  }
  // broadcast via smem[0]
  if (threadIdx.x == 0) smem[0] = out;
  __syncthreads();
  return smem[0];
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

__global__ void softmax_lastdim_f32_kernel(const float* __restrict__ x,
                                          float* __restrict__ y,
                                          int64_t rows,
                                          int64_t cols) {
  const int64_t row = (int64_t)blockIdx.x;
  if (row >= rows) return;

  const int64_t base = row * cols;

  // 1) row max
  float tmax = -INFINITY;
  for (int64_t c = (int64_t)threadIdx.x; c < cols; c += (int64_t)blockDim.x) {
    float v = x[base + c];
    tmax = (v > tmax) ? v : tmax;
  }
  const float rmax = block_reduce_max(tmax);

  // 2) sum exp
  float tsum = 0.0f;
  for (int64_t c = (int64_t)threadIdx.x; c < cols; c += (int64_t)blockDim.x) {
    float v = x[base + c];
    tsum += __expf(v - rmax);
  }
  const float rsum = block_reduce_sum(tsum);

  // 3) write
  const float inv = 1.0f / rsum;
  for (int64_t c = (int64_t)threadIdx.x; c < cols; c += (int64_t)blockDim.x) {
    float v = x[base + c];
    y[base + c] = __expf(v - rmax) * inv;
  }
}

__global__ void softmax_lastdim_f16_kernel(const __half* __restrict__ x,
                                          __half* __restrict__ y,
                                          int64_t rows,
                                          int64_t cols) {
  const int64_t row = (int64_t)blockIdx.x;
  if (row >= rows) return;

  const int64_t base = row * cols;

  float tmax = -INFINITY;
  for (int64_t c = (int64_t)threadIdx.x; c < cols; c += (int64_t)blockDim.x) {
    float v = __half2float(x[base + c]);
    tmax = (v > tmax) ? v : tmax;
  }
  const float rmax = block_reduce_max(tmax);

  float tsum = 0.0f;
  for (int64_t c = (int64_t)threadIdx.x; c < cols; c += (int64_t)blockDim.x) {
    float v = __half2float(x[base + c]);
    tsum += __expf(v - rmax);
  }
  const float rsum = block_reduce_sum(tsum);

  const float inv = 1.0f / rsum;
  for (int64_t c = (int64_t)threadIdx.x; c < cols; c += (int64_t)blockDim.x) {
    float v = __half2float(x[base + c]);
    float out = __expf(v - rmax) * inv;
    y[base + c] = __float2half_rn(out);
  }
}

} // namespace softmax_impl

// ============================================================================
// Variant: Softmax f32 (axis = last dim)
// inputs=(X), outputs=(Y)
// forbids in-place: Y must NOT alias X.
// ============================================================================
static inline bool softmax_check_f32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& X = inputs[0];
  const TensorDesc& Y = outputs[0];

  if (!is_f32_contig_anyrank(X) || !is_f32_contig_anyrank(Y)) return false;
  if (!same_shape(X, Y)) return false;

  int64_t numel = 0;
  if (!compute_numel(X, &numel)) return false;
  if (numel <= 0) return false;

  const int64_t cols = last_dim(X);
  if (cols <= 0) return false;

  // rows must be integer
  if ((numel % cols) != 0) return false;

  return true;
}

static bool softmax_supported_f32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return softmax_check_f32(inputs, num_inputs, outputs, num_outputs);
}

static Status softmax_launch_f32(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void*, size_t,
    cudaStream_t stream) {

  if (!softmax_check_f32(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& X = inputs[0];
  TensorDesc& Y = outputs[0];

  // forbid in-place (softmax needs original X multiple times)
  if (Y.data == X.data) return Status::InvalidArgument;

  int64_t numel = 0;
  (void)compute_numel(X, &numel);

  const int64_t cols = last_dim(X);
  const int64_t rows = numel / cols;

  // launch: one block per row
  // threads: fixed 256 (괜찮은 기본값)
  constexpr int kThreads = 256;
  int blocks = (rows > 0) ? (int)rows : 1;
  // grid.x max clamp
  if (blocks > 2147483647) blocks = 2147483647;

  cudaGetLastError(); // clear
  softmax_impl::softmax_lastdim_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)X.data, (float*)Y.data, rows, cols);

  return cuda_last_status();
}

KernelVariant make_softmax_f32_variant() {
  KernelVariant v{};
  v.name = "softmax_lastdim_f32";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = 0; // no attr for now
  v.launch = softmax_launch_f32;
  v.supported = softmax_supported_f32;
  v.query_workspace = softmax_workspace;
  return v;
}

// ============================================================================
// Variant: Softmax f16 (axis = last dim)
// ============================================================================
static inline bool softmax_check_f16(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& X = inputs[0];
  const TensorDesc& Y = outputs[0];

  if (!is_f16_contig_anyrank(X) || !is_f16_contig_anyrank(Y)) return false;
  if (!same_shape(X, Y)) return false;

  int64_t numel = 0;
  if (!compute_numel(X, &numel)) return false;
  if (numel <= 0) return false;

  const int64_t cols = last_dim(X);
  if (cols <= 0) return false;
  if ((numel % cols) != 0) return false;

  return true;
}

static bool softmax_supported_f16(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return softmax_check_f16(inputs, num_inputs, outputs, num_outputs);
}

static Status softmax_launch_f16(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void*, size_t,
    cudaStream_t stream) {

  if (!softmax_check_f16(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& X = inputs[0];
  TensorDesc& Y = outputs[0];

  if (Y.data == X.data) return Status::InvalidArgument;

  int64_t numel = 0;
  (void)compute_numel(X, &numel);

  const int64_t cols = last_dim(X);
  const int64_t rows = numel / cols;

  constexpr int kThreads = 256;
  int blocks = (rows > 0) ? (int)rows : 1;
  if (blocks > 2147483647) blocks = 2147483647;

  cudaGetLastError(); // clear
  softmax_impl::softmax_lastdim_f16_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half*)X.data, (__half*)Y.data, rows, cols);

  return cuda_last_status();
}

KernelVariant make_softmax_f16_variant() {
  KernelVariant v{};
  v.name = "softmax_lastdim_f16";
  v.priority = 10;
  v.flags = 0;
  v.expected_attr_schema_id = 0; // no attr for now
  v.launch = softmax_launch_f16;
  v.supported = softmax_supported_f16;
  v.query_workspace = softmax_workspace;
  return v;
}

} // namespace aicf::cuda
