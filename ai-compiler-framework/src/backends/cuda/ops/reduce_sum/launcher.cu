#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API
#include <aicf/backends/cuda/ops/reduce_sum/api.hpp>

// registry glue
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_pack.hpp>

// common shim
#include "aicf/backends/cuda/ops/_common/shim/launch.hpp"
#include "aicf/backends/cuda/ops/_common/shim/status.hpp"
#include "aicf/backends/cuda/ops/_common/shim/validate.hpp"

#include "kernels.cuh"

#include <string_view>

namespace aicf::cuda {

// -------------------------
// warp reduce helper (device)
// -------------------------
static __inline__ __device__ float warp_sum(float v) {
  // full mask
  unsigned mask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(mask, v, offset);
  }
  return v;
}

// -------------------------
// kernels (definitions)
// -------------------------
namespace reduce_sum_impl {

// f32 -> f32
__global__ void reduce_sum_lastdim_f32_kernel(const float* __restrict__ dY,
                                             float* __restrict__ dB,
                                             int M, int N) {
  const int col = (int)blockIdx.x;
  if (col >= N) return;

  float acc = 0.0f;

  // stride over rows
  for (int row = (int)threadIdx.x; row < M; row += (int)blockDim.x) {
    acc += dY[(int64_t)row * N + col];
  }

  // warp reduce
  acc = warp_sum(acc);

  // one atomic per warp
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&dB[col], acc);
  }
}

// f16 -> f32 (accumulate in float)
__global__ void reduce_sum_lastdim_f16_kernel(const __half* __restrict__ dY,
                                             float* __restrict__ dB,
                                             int M, int N) {
  const int col = (int)blockIdx.x;
  if (col >= N) return;

  float acc = 0.0f;

  for (int row = (int)threadIdx.x; row < M; row += (int)blockDim.x) {
    acc += __half2float(dY[(int64_t)row * N + col]);
  }

  acc = warp_sum(acc);

  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&dB[col], acc);
  }
}

__global__ void reduce_sum_lastdim_f16x2_kernel(
    const __half2* __restrict__ dY,
    float* __restrict__ dB,
    int M, int N2)
{
  const int col2 = (int)blockIdx.x;
  if (col2 >= N2) return;

  float acc0 = 0.f, acc1 = 0.f;

  for (int row = (int)threadIdx.x; row < M; row += (int)blockDim.x) {
    __half2 v = dY[(int64_t)row * N2 + col2];
    acc0 += __half2float(__low2half(v));
    acc1 += __half2float(__high2half(v));
  }

  acc0 = warp_sum(acc0);
  acc1 = warp_sum(acc1);

  if ((threadIdx.x & 31) == 0) {
    const int base = col2 * 2;
    atomicAdd(&dB[base + 0], acc0);
    atomicAdd(&dB[base + 1], acc1);
  }
}


} // namespace reduce_sum_impl

// -------------------------
// public API implementation (keep existing)
// -------------------------
aicf::Status reduce_sum_lastdim_f32(const float* dY,
                                   float* dB,
                                   int M, int N,
                                   aicf::Stream stream) {
  if (!dY || !dB || M <= 0 || N <= 0) {
    return aicf::Status::InvalidArgument;
  }

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  // IMPORTANT: dB must be zeroed before atomic reduction
  cudaError_t e = cudaMemsetAsync(dB, 0, (size_t)N * sizeof(float), s);
  if (e != cudaSuccess) return aicf::cuda::shim::cuda_last_error_to_status();

  constexpr int kThreads = 256;
  dim3 block(kThreads, 1, 1);
  dim3 grid(N, 1, 1);

  reduce_sum_impl::reduce_sum_lastdim_f32_kernel<<<grid, block, 0, s>>>(dY, dB, M, N);
  return aicf::cuda::shim::cuda_last_error_to_status();
}

// -------------------------
// Attr helpers (same style)
// -------------------------
static inline bool attr_get_i64(const void* attr, const char* key, int64_t* out_val) {
  if (!attr) return false;
  const auto* pack = static_cast<const aicf::cuda::AttrPack*>(attr);
  if (!pack->items || pack->size <= 0) return false;

  const std::string_view k(key);
  for (int i = 0; i < pack->size; ++i) {
    const auto& kv = pack->items[i];
    if (!kv.key) continue;
    if (std::string_view(kv.key) == k) {
      if (kv.val.tag == aicf::cuda::AttrTag::kI64) {
        *out_val = kv.val.i64;
        return true;
      }
      return false;
    }
  }
  return false;
}

// -------------------------
// last-dim-only helpers (safe)
// -------------------------
static inline bool is_f32_contig_rank_ge2(const TensorDesc& T) {
  return (T.dtype == DType::kF32) && T.contiguous && (T.rank() >= 2);
}

static inline bool is_f16_contig_rank_ge2(const TensorDesc& T) {
  return (T.dtype == DType::kF16) && T.contiguous && (T.rank() >= 2);
}

static inline bool axis_is_last_dim_only(const TensorDesc& Y, int64_t axis_raw) {
  const int64_t r = Y.rank();
  if (r < 2) return false;
  const int64_t last = r - 1;
  if (axis_raw == -1) return true;
  if (axis_raw == last) return true;
  return false;
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

// -------------------------
// Checks
// - F32 path: dY f32 -> dB f32
// - F16 path: dY f16 -> dB f32  (accumulate f32)
// -------------------------
static inline bool reduce_sum_check_f32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    int64_t axis_raw) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& dY = inputs[0];
  const TensorDesc& dB = outputs[0];

  if (!is_f32_contig_rank_ge2(dY)) return false;
  if (!(dB.dtype == DType::kF32 && dB.contiguous && dB.rank() == 1)) return false;

  if (!axis_is_last_dim_only(dY, axis_raw)) return false;

  int64_t M = 0, N = 0;
  if (!compute_MN_last_dim(dY, &M, &N)) return false;
  return (dB.shape[0] == N);
}

static inline bool reduce_sum_check_f16_to_f32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    int64_t axis_raw) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& dY = inputs[0];
  const TensorDesc& dB = outputs[0];

  if (!is_f16_contig_rank_ge2(dY)) return false;
  if (!(dB.dtype == DType::kF32 && dB.contiguous && dB.rank() == 1)) return false;

  if (!axis_is_last_dim_only(dY, axis_raw)) return false;

  int64_t M = 0, N = 0;
  if (!compute_MN_last_dim(dY, &M, &N)) return false;
  return (dB.shape[0] == N);
}

// vec2 eligibility for f16 input (N even + 4B align on dY and dB)
static inline bool reduce_sum_f16_vec2_ok(const TensorDesc& dY, const TensorDesc& dB) {
  const int64_t r = dY.rank();
  const int64_t N = dY.shape[r - 1];
  if ((N & 1) != 0) return false;

  constexpr size_t kAlign = 4;
  if (!aicf::cuda::shim::is_aligned_data(dY, kAlign)) return false;
  if (!aicf::cuda::shim::is_aligned_data(dB, kAlign)) return false; // dB is float*, 4B align usually true
  return true;
}

static size_t reduce_sum_workspace(const TensorDesc*, int, const void*) { return 0; }

// -------------------------
// Variant: F32
// -------------------------
static bool reduce_sum_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  int64_t axis = -1;
  (void)attr_get_i64(attr, "axis", &axis);
  return reduce_sum_check_f32(inputs, num_inputs, outputs, num_outputs, axis);
}

static aicf::Status reduce_sum_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  int64_t axis = -1;
  (void)attr_get_i64(attr, "axis", &axis);

  if (!reduce_sum_check_f32(inputs, num_inputs, outputs, num_outputs, axis)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& dY = inputs[0];
  TensorDesc& dB = outputs[0];

  int64_t M64 = 0, N64 = 0;
  if (!compute_MN_last_dim(dY, &M64, &N64)) return aicf::Status::InvalidArgument;

  const int M = (int)M64;
  const int N = (int)N64;

  // zero init for atomic reduction
  cudaError_t e = cudaMemsetAsync(dB.data, 0, (size_t)N * sizeof(float), stream);
  if (e != cudaSuccess) return aicf::cuda::shim::cuda_last_error_to_status();

  constexpr int kThreads = 256;
  dim3 block(kThreads, 1, 1);
  dim3 grid(N, 1, 1);

  reduce_sum_impl::reduce_sum_lastdim_f32_kernel<<<grid, block, 0, stream>>>(
      (const float*)dY.data, (float*)dB.data, M, N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_reduce_sum_lastdim_f32_variant() {
  KernelVariant v{};
  v.name = "reduce_sum_lastdim_f32";
  v.priority = 0;
  v.flags = 0;
  v.launch = reduce_sum_f32_launch;
  v.supported = reduce_sum_f32_supported;
  v.query_workspace = reduce_sum_workspace;
  return v;
}

// -------------------------
// Variant: F16 input -> F32 output (naive)
// -------------------------
static bool reduce_sum_f16_to_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  int64_t axis = -1;
  (void)attr_get_i64(attr, "axis", &axis);
  return reduce_sum_check_f16_to_f32(inputs, num_inputs, outputs, num_outputs, axis);
}

static aicf::Status reduce_sum_f16_to_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  int64_t axis = -1;
  (void)attr_get_i64(attr, "axis", &axis);

  if (!reduce_sum_check_f16_to_f32(inputs, num_inputs, outputs, num_outputs, axis)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& dY = inputs[0];
  TensorDesc& dB = outputs[0];

  int64_t M64 = 0, N64 = 0;
  if (!compute_MN_last_dim(dY, &M64, &N64)) return aicf::Status::InvalidArgument;

  const int M = (int)M64;
  const int N = (int)N64;

  // zero init
  cudaError_t e = cudaMemsetAsync(dB.data, 0, (size_t)N * sizeof(float), stream);
  if (e != cudaSuccess) return aicf::cuda::shim::cuda_last_error_to_status();

  constexpr int kThreads = 256;

  // half2 fastpath if possible (still outputs float[N])
  if (reduce_sum_f16_vec2_ok(dY, dB)) {
    const int N2 = N / 2;
    dim3 block(kThreads, 1, 1);
    dim3 grid(N2, 1, 1);

    reduce_sum_impl::reduce_sum_lastdim_f16x2_kernel<<<grid, block, 0, stream>>>(
        (const __half2*)dY.data, (float*)dB.data, M, N2);
  } else {
    dim3 block(kThreads, 1, 1);
    dim3 grid(N, 1, 1);

    reduce_sum_impl::reduce_sum_lastdim_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)dY.data, (float*)dB.data, M, N);
  }

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_reduce_sum_lastdim_f16_to_f32_variant() {
  KernelVariant v{};
  v.name = "reduce_sum_lastdim_f16_to_f32";
  v.priority = 0;
  v.flags = 0;
  v.launch = reduce_sum_f16_to_f32_launch;
  v.supported = reduce_sum_f16_to_f32_supported;
  v.query_workspace = reduce_sum_workspace;
  return v;
}

} // namespace aicf::cuda
