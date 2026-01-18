#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <climits>
#include <cstring>

#include <aicf/backends/cuda/ops/reduce_sum/api.hpp> // ⚠️ 이 파일도 아래 설명대로 바꿔야 함

#include <aicf/backends/cuda/registry/status.hpp>
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_blob.hpp>

#include "kernels.cuh"

namespace aicf::cuda {

// -------------------------
// small helpers (core-free)
// -------------------------
static inline Status cuda_err_to_status(cudaError_t e) {
  return (e == cudaSuccess) ? Status::Ok : Status::Internal;
}

static inline Status cuda_last_error_to_status() {
  // launch error
  cudaError_t e = cudaPeekAtLastError();
  if (e != cudaSuccess) return Status::Internal;
  return Status::Ok;
}

static inline bool is_aligned_ptr(const void* p, size_t align) {
  return ((reinterpret_cast<uintptr_t>(p) & (align - 1)) == 0);
}

static inline bool is_contig_rank_ge2(const TensorDesc& T) {
  return T.contiguous && (T.rank() >= 2);
}

static inline bool compute_MN_keep_lastdim(const TensorDesc& Y, int64_t* out_M, int64_t* out_N) {
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

static inline bool out_is_f32_contig_vecN(const TensorDesc& dB, int64_t N) {
  return (dB.dtype == DType::kF32) && dB.contiguous && (dB.rank() == 1) && (dB.shape[0] == N);
}

// -------------------------
// warp reduce helper (device)
// -------------------------
static __inline__ __device__ float warp_sum(float v) {
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

__global__ void reduce_sum_rows_f32_kernel(const float* __restrict__ dY,
                                          float* __restrict__ dB,
                                          int M, int N) {
  const int col = (int)blockIdx.x;
  if (col >= N) return;

  float acc = 0.0f;
  for (int row = (int)threadIdx.x; row < M; row += (int)blockDim.x) {
    acc += dY[(int64_t)row * N + col];
  }

  acc = warp_sum(acc);

  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&dB[col], acc);
  }
}

__global__ void reduce_sum_rows_f16_kernel(const __half* __restrict__ dY,
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

__global__ void reduce_sum_rows_f16x2_kernel(const __half2* __restrict__ dY,
                                            float* __restrict__ dB,
                                            int M, int N2) {
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

// ============================================================================
// AttrBlob schema (ReduceSum)
// ============================================================================
static constexpr uint32_t kAttrSchema_ReduceSum = 0x5253554Du; // 'RSUM'
struct ReduceSumAttrV0 { int64_t axis; };

static inline int64_t get_axis_default0(const void* attr) {
  const AttrBlob* ab = static_cast<const AttrBlob*>(attr);
  if (!ab) return 0;

  if (ab->schema_id != 0 && ab->schema_id != kAttrSchema_ReduceSum) return 0;
  if (!ab->data || ab->bytes < sizeof(ReduceSumAttrV0)) return 0;

  const auto* a = static_cast<const ReduceSumAttrV0*>(ab->data);
  return a->axis;
}

static inline bool reduce_sum_rows_check_f32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    int64_t axis) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& dY = inputs[0];
  const TensorDesc& dB = outputs[0];

  if (dY.dtype != DType::kF32) return false;
  if (!is_contig_rank_ge2(dY)) return false;
  if (axis != 0) return false;

  int64_t M = 0, N = 0;
  if (!compute_MN_keep_lastdim(dY, &M, &N)) return false;
  return out_is_f32_contig_vecN(dB, N);
}

static inline bool reduce_sum_rows_check_f16_to_f32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    int64_t axis) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& dY = inputs[0];
  const TensorDesc& dB = outputs[0];

  if (dY.dtype != DType::kF16) return false;
  if (!is_contig_rank_ge2(dY)) return false;
  if (axis != 0) return false;

  int64_t M = 0, N = 0;
  if (!compute_MN_keep_lastdim(dY, &M, &N)) return false;
  return out_is_f32_contig_vecN(dB, N);
}

// vec2 eligibility: N even + 4B align on dY and dB
static inline bool reduce_sum_f16_vec2_ok(const TensorDesc& dY, const TensorDesc& dB) {
  const int64_t r = dY.rank();
  const int64_t N = dY.shape[r - 1];
  if ((N & 1) != 0) return false;

  constexpr size_t kAlign = 4;
  if (!is_aligned_ptr(dY.data, kAlign)) return false;
  if (!is_aligned_ptr(dB.data, kAlign)) return false;
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

  const int64_t axis = get_axis_default0(attr);
  return reduce_sum_rows_check_f32(inputs, num_inputs, outputs, num_outputs, axis);
}

static Status reduce_sum_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  const int64_t axis = get_axis_default0(attr);
  if (!reduce_sum_rows_check_f32(inputs, num_inputs, outputs, num_outputs, axis))
    return Status::InvalidArgument;

  const TensorDesc& dY = inputs[0];
  TensorDesc& dB = outputs[0];

  int64_t M64 = 0, N64 = 0;
  if (!compute_MN_keep_lastdim(dY, &M64, &N64)) return Status::InvalidArgument;
  if (M64 > INT_MAX || N64 > INT_MAX) return Status::NotImplemented;

  const int M = (int)M64;
  const int N = (int)N64;

  cudaError_t e = cudaMemsetAsync(dB.data, 0, (size_t)N * sizeof(float), stream);
  if (e != cudaSuccess) return cuda_err_to_status(e);

  constexpr int kThreads = 256;
  dim3 block(kThreads, 1, 1);
  dim3 grid(N, 1, 1);

  reduce_sum_impl::reduce_sum_rows_f32_kernel<<<grid, block, 0, stream>>>(
      (const float*)dY.data, (float*)dB.data, M, N);

  return cuda_last_error_to_status();
}

KernelVariant make_reduce_sum_lastdim_f32_variant() {
  KernelVariant v{};
  v.name = "reduce_sum_keep_lastdim_f32";
  v.priority = 100;
  v.flags = 0;
  v.launch = reduce_sum_f32_launch;
  v.supported = reduce_sum_f32_supported;
  v.query_workspace = reduce_sum_workspace;
  // v.expected_attr_schema_id = kAttrSchema_ReduceSum;
  return v;
}

// -------------------------
// Variant: F16 -> F32
// -------------------------
static bool reduce_sum_f16_to_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  const int64_t axis = get_axis_default0(attr);
  return reduce_sum_rows_check_f16_to_f32(inputs, num_inputs, outputs, num_outputs, axis);
}

static Status reduce_sum_f16_to_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  const int64_t axis = get_axis_default0(attr);
  if (!reduce_sum_rows_check_f16_to_f32(inputs, num_inputs, outputs, num_outputs, axis))
    return Status::InvalidArgument;

  const TensorDesc& dY = inputs[0];
  TensorDesc& dB = outputs[0];

  int64_t M64 = 0, N64 = 0;
  if (!compute_MN_keep_lastdim(dY, &M64, &N64)) return Status::InvalidArgument;
  if (M64 > INT_MAX || N64 > INT_MAX) return Status::NotImplemented;

  const int M = (int)M64;
  const int N = (int)N64;

  cudaError_t e = cudaMemsetAsync(dB.data, 0, (size_t)N * sizeof(float), stream);
  if (e != cudaSuccess) return cuda_err_to_status(e);

  constexpr int kThreads = 256;

  if (reduce_sum_f16_vec2_ok(dY, dB)) {
    const int N2 = N / 2;
    dim3 block(kThreads, 1, 1);
    dim3 grid(N2, 1, 1);

    reduce_sum_impl::reduce_sum_rows_f16x2_kernel<<<grid, block, 0, stream>>>(
        (const __half2*)dY.data, (float*)dB.data, M, N2);
  } else {
    dim3 block(kThreads, 1, 1);
    dim3 grid(N, 1, 1);

    reduce_sum_impl::reduce_sum_rows_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)dY.data, (float*)dB.data, M, N);
  }

  return cuda_last_error_to_status();
}

KernelVariant make_reduce_sum_lastdim_f16_to_f32_variant() {
  KernelVariant v{};
  v.name = "reduce_sum_keep_lastdim_f16_to_f32";
  v.priority = 110;
  v.flags = 0;
  v.launch = reduce_sum_f16_to_f32_launch;
  v.supported = reduce_sum_f16_to_f32_supported;
  v.query_workspace = reduce_sum_workspace;
  // v.expected_attr_schema_id = kAttrSchema_ReduceSum;
  return v;
}

// ============================================================================
// public API (core-free 버전)
//   ⚠️ api.hpp도 반드시 이 시그니처로 맞춰야 함.
// ============================================================================
Status reduce_sum_lastdim_f32(const float* dY, float* dB, int M, int N, cudaStream_t stream) {
  if (!dY || !dB || M <= 0 || N <= 0) return Status::InvalidArgument;

  cudaError_t e = cudaMemsetAsync(dB, 0, (size_t)N * sizeof(float), stream);
  if (e != cudaSuccess) return cuda_err_to_status(e);

  constexpr int kThreads = 256;
  dim3 block(kThreads, 1, 1);
  dim3 grid(N, 1, 1);

  reduce_sum_impl::reduce_sum_rows_f32_kernel<<<grid, block, 0, stream>>>(dY, dB, M, N);
  return cuda_last_error_to_status();
}

} // namespace aicf::cuda
