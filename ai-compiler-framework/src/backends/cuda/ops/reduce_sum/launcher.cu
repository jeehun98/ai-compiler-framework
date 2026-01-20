#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <climits>
#include <cstring>

#include <aicf/backends/cuda/ops/reduce_sum/api.hpp> // ⚠️ 아래 api.hpp도 이 시그니처로 맞춤

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
  cudaError_t e = cudaPeekAtLastError();
  return (e == cudaSuccess) ? Status::Ok : Status::Internal;
}

static inline bool is_aligned_ptr(const void* p, size_t align) {
  return ((reinterpret_cast<uintptr_t>(p) & (align - 1)) == 0);
}

static inline bool is_contig_rank_ge2(const TensorDesc& T) {
  return T.contiguous && (T.rank() >= 2);
}

// flatten leading dims into M, keep last dim as N
static inline bool compute_MN_keep_lastdim(const TensorDesc& dY, int64_t* out_M, int64_t* out_N) {
  const int64_t r = dY.rank();
  if (r < 2) return false;

  const int64_t N = dY.shape[r - 1];
  if (N <= 0) return false;

  int64_t M = 1;
  for (int64_t i = 0; i < r - 1; ++i) {
    const int64_t d = dY.shape[i];
    if (d <= 0) return false;
    M *= d;
  }

  if (M <= 0) return false;
  *out_M = M;
  *out_N = N;
  return true;
}

static inline bool out_is_f32_contig_vecN(const TensorDesc& out, int64_t N) {
  return (out.dtype == DType::kF32) && out.contiguous && (out.rank() == 1) && (out.shape[0] == N);
}

static inline bool out_is_f16_contig_vecN(const TensorDesc& out, int64_t N) {
  return (out.dtype == DType::kF16) && out.contiguous && (out.rank() == 1) && (out.shape[0] == N);
}

// -------------------------
// AttrBlob schema (ReduceSum)
//   NOTE: 이 커널은 "keep_lastdim" 형태만 구현:
//     dY: [..., N] -> out: [N]
//   따라서 axis 의미는 다음으로 해석:
//     axis==0 : keep_lastdim 형태로 reduction 수행 (현재 v2 lower가 axis=0으로 내려옴)
//   (axis=-1, axis=lastdim 등은 여기서 쓰지 않는 걸 권장: kernel_id로 박제)
// -------------------------
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

// vec2 eligibility: N even + 4B align
static inline bool reduce_sum_f16_vec2_ok_inout_f32(const TensorDesc& dY, const TensorDesc& out_f32) {
  const int64_t r = dY.rank();
  const int64_t N = dY.shape[r - 1];
  if ((N & 1) != 0) return false;
  constexpr size_t kAlign = 4;
  if (!is_aligned_ptr(dY.data, kAlign)) return false;
  if (!is_aligned_ptr(out_f32.data, kAlign)) return false;
  return true;
}

static inline bool reduce_sum_f16_vec2_ok_inout_f16(const TensorDesc& dY, const TensorDesc& out_f16) {
  const int64_t r = dY.rank();
  const int64_t N = dY.shape[r - 1];
  if ((N & 1) != 0) return false;
  constexpr size_t kAlign = 4;
  if (!is_aligned_ptr(dY.data, kAlign)) return false;
  if (!is_aligned_ptr(out_f16.data, kAlign)) return false;
  return true;
}

// ------------------------------------------------------------
// kernels (definitions) - block reduction (no atomicAdd)
// ------------------------------------------------------------
namespace reduce_sum_impl {

static __inline__ __device__ float warp_sum(float v) {
  unsigned mask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(mask, v, offset);
  return v;
}

template <int kThreads>
__device__ __forceinline__ float block_sum(float v) {
  // warp reduce
  v = warp_sum(v);

  __shared__ float smem[32]; // up to 1024 threads -> 32 warps
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;
  if (lane == 0) smem[wid] = v;
  __syncthreads();

  // first warp reduce warp-sums
  float out = 0.0f;
  if (wid == 0) {
    int nwarps = (kThreads + 31) / 32;
    out = (lane < nwarps) ? smem[lane] : 0.0f;
    out = warp_sum(out);
  }
  return out; // valid on wid==0 lane==0
}

__global__ void reduce_sum_rows_f32_to_f32_kernel(
    const float* __restrict__ dY,
    float* __restrict__ out,
    int M, int N) {

  const int col = (int)blockIdx.x;
  if (col >= N) return;

  float acc = 0.0f;
  for (int row = (int)threadIdx.x; row < M; row += (int)blockDim.x) {
    acc += dY[(int64_t)row * N + col];
  }

  const float total = block_sum<256>(acc);
  if (threadIdx.x == 0) out[col] = total;
}

__global__ void reduce_sum_rows_f16_to_f32_kernel(
    const __half* __restrict__ dY,
    float* __restrict__ out,
    int M, int N) {

  const int col = (int)blockIdx.x;
  if (col >= N) return;

  float acc = 0.0f;
  for (int row = (int)threadIdx.x; row < M; row += (int)blockDim.x) {
    acc += __half2float(dY[(int64_t)row * N + col]);
  }

  const float total = block_sum<256>(acc);
  if (threadIdx.x == 0) out[col] = total;
}

__global__ void reduce_sum_rows_f16x2_to_f32_kernel(
    const __half2* __restrict__ dY,
    float* __restrict__ out,
    int M, int N2) {

  const int col2 = (int)blockIdx.x;
  if (col2 >= N2) return;

  float acc0 = 0.f, acc1 = 0.f;
  for (int row = (int)threadIdx.x; row < M; row += (int)blockDim.x) {
    __half2 v = dY[(int64_t)row * N2 + col2];
    acc0 += __half2float(__low2half(v));
    acc1 += __half2float(__high2half(v));
  }

  const float t0 = block_sum<256>(acc0);
  const float t1 = block_sum<256>(acc1);

  if (threadIdx.x == 0) {
    const int base = col2 * 2;
    out[base + 0] = t0;
    out[base + 1] = t1;
  }
}

__global__ void reduce_sum_rows_f16_to_f16_kernel(
    const __half* __restrict__ dY,
    __half* __restrict__ out,
    int M, int N) {

  const int col = (int)blockIdx.x;
  if (col >= N) return;

  float acc = 0.0f;
  for (int row = (int)threadIdx.x; row < M; row += (int)blockDim.x) {
    acc += __half2float(dY[(int64_t)row * N + col]);
  }

  const float total = block_sum<256>(acc);
  if (threadIdx.x == 0) out[col] = __float2half(total);
}

__global__ void reduce_sum_rows_f16x2_to_f16_kernel(
    const __half2* __restrict__ dY,
    __half2* __restrict__ out2,
    int M, int N2) {

  const int col2 = (int)blockIdx.x;
  if (col2 >= N2) return;

  float acc0 = 0.f, acc1 = 0.f;
  for (int row = (int)threadIdx.x; row < M; row += (int)blockDim.x) {
    __half2 v = dY[(int64_t)row * N2 + col2];
    acc0 += __half2float(__low2half(v));
    acc1 += __half2float(__high2half(v));
  }

  const float t0 = block_sum<256>(acc0);
  const float t1 = block_sum<256>(acc1);

  if (threadIdx.x == 0) {
    out2[col2] = __halves2half2(__float2half(t0), __float2half(t1));
  }
}

} // namespace reduce_sum_impl

// ------------------------------------------------------------
// checks
// ------------------------------------------------------------
static inline bool reduce_sum_keep_lastdim_check_f32_to_f32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    int64_t axis) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& dY = inputs[0];
  const TensorDesc& out = outputs[0];

  if (dY.dtype != DType::kF32) return false;
  if (!is_contig_rank_ge2(dY)) return false;

  // current contract: axis must be 0 (as your lower emits)
  if (axis != 0) return false;

  int64_t M = 0, N = 0;
  if (!compute_MN_keep_lastdim(dY, &M, &N)) return false;
  return out_is_f32_contig_vecN(out, N);
}

static inline bool reduce_sum_keep_lastdim_check_f16_to_f32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    int64_t axis) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& dY = inputs[0];
  const TensorDesc& out = outputs[0];

  if (dY.dtype != DType::kF16) return false;
  if (!is_contig_rank_ge2(dY)) return false;
  if (axis != 0) return false;

  int64_t M = 0, N = 0;
  if (!compute_MN_keep_lastdim(dY, &M, &N)) return false;
  return out_is_f32_contig_vecN(out, N);
}

static inline bool reduce_sum_keep_lastdim_check_f16_to_f16(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    int64_t axis) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& dY = inputs[0];
  const TensorDesc& out = outputs[0];

  if (dY.dtype != DType::kF16) return false;
  if (!is_contig_rank_ge2(dY)) return false;
  if (axis != 0) return false;

  int64_t M = 0, N = 0;
  if (!compute_MN_keep_lastdim(dY, &M, &N)) return false;
  return out_is_f16_contig_vecN(out, N);
}

static size_t reduce_sum_workspace(const TensorDesc*, int, const void*) { return 0; }

// ------------------------------------------------------------
// Variant: F32 -> F32
// ------------------------------------------------------------
static bool reduce_sum_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  const int64_t axis = get_axis_default0(attr);
  return reduce_sum_keep_lastdim_check_f32_to_f32(inputs, num_inputs, outputs, num_outputs, axis);
}

static Status reduce_sum_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  const int64_t axis = get_axis_default0(attr);
  if (!reduce_sum_keep_lastdim_check_f32_to_f32(inputs, num_inputs, outputs, num_outputs, axis))
    return Status::InvalidArgument;

  const TensorDesc& dY = inputs[0];
  TensorDesc& out = outputs[0];

  int64_t M64 = 0, N64 = 0;
  if (!compute_MN_keep_lastdim(dY, &M64, &N64)) return Status::InvalidArgument;
  if (M64 > INT_MAX || N64 > INT_MAX) return Status::NotImplemented;

  const int M = (int)M64;
  const int N = (int)N64;

  constexpr int kThreads = 256;
  dim3 block(kThreads, 1, 1);
  dim3 grid(N, 1, 1);

  reduce_sum_impl::reduce_sum_rows_f32_to_f32_kernel<<<grid, block, 0, stream>>>(
      (const float*)dY.data, (float*)out.data, M, N);

  return cuda_last_error_to_status();
}

KernelVariant make_reduce_sum_lastdim_f32_variant() {
  KernelVariant v{};
  v.name = "reduce_sum_keep_lastdim_f32_to_f32";
  v.priority = 100;
  v.flags = 0;
  v.launch = reduce_sum_f32_launch;
  v.supported = reduce_sum_f32_supported;
  v.query_workspace = reduce_sum_workspace;
  return v;
}

// ------------------------------------------------------------
// Variant: F16 -> F32 (existing id 유지 가능)
// ------------------------------------------------------------
static bool reduce_sum_f16_to_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  const int64_t axis = get_axis_default0(attr);
  return reduce_sum_keep_lastdim_check_f16_to_f32(inputs, num_inputs, outputs, num_outputs, axis);
}

static Status reduce_sum_f16_to_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  const int64_t axis = get_axis_default0(attr);
  if (!reduce_sum_keep_lastdim_check_f16_to_f32(inputs, num_inputs, outputs, num_outputs, axis))
    return Status::InvalidArgument;

  const TensorDesc& dY = inputs[0];
  TensorDesc& out = outputs[0];

  int64_t M64 = 0, N64 = 0;
  if (!compute_MN_keep_lastdim(dY, &M64, &N64)) return Status::InvalidArgument;
  if (M64 > INT_MAX || N64 > INT_MAX) return Status::NotImplemented;

  const int M = (int)M64;
  const int N = (int)N64;

  constexpr int kThreads = 256;

  if (reduce_sum_f16_vec2_ok_inout_f32(dY, out)) {
    const int N2 = N / 2;
    dim3 block(kThreads, 1, 1);
    dim3 grid(N2, 1, 1);

    reduce_sum_impl::reduce_sum_rows_f16x2_to_f32_kernel<<<grid, block, 0, stream>>>(
        (const __half2*)dY.data, (float*)out.data, M, N2);
  } else {
    dim3 block(kThreads, 1, 1);
    dim3 grid(N, 1, 1);

    reduce_sum_impl::reduce_sum_rows_f16_to_f32_kernel<<<grid, block, 0, stream>>>(
        (const __half*)dY.data, (float*)out.data, M, N);
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
  return v;
}

// ------------------------------------------------------------
// ✅ NEW Variant: F16 -> F16  (너 지금 db가 f16이라 이게 필요)
// ------------------------------------------------------------
static bool reduce_sum_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  const int64_t axis = get_axis_default0(attr);
  return reduce_sum_keep_lastdim_check_f16_to_f16(inputs, num_inputs, outputs, num_outputs, axis);
}

static Status reduce_sum_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  const int64_t axis = get_axis_default0(attr);
  if (!reduce_sum_keep_lastdim_check_f16_to_f16(inputs, num_inputs, outputs, num_outputs, axis))
    return Status::InvalidArgument;

  const TensorDesc& dY = inputs[0];
  TensorDesc& out = outputs[0];

  int64_t M64 = 0, N64 = 0;
  if (!compute_MN_keep_lastdim(dY, &M64, &N64)) return Status::InvalidArgument;
  if (M64 > INT_MAX || N64 > INT_MAX) return Status::NotImplemented;

  const int M = (int)M64;
  const int N = (int)N64;

  constexpr int kThreads = 256;

  if (reduce_sum_f16_vec2_ok_inout_f16(dY, out)) {
    const int N2 = N / 2;
    dim3 block(kThreads, 1, 1);
    dim3 grid(N2, 1, 1);

    reduce_sum_impl::reduce_sum_rows_f16x2_to_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half2*)dY.data, (__half2*)out.data, M, N2);
  } else {
    dim3 block(kThreads, 1, 1);
    dim3 grid(N, 1, 1);

    reduce_sum_impl::reduce_sum_rows_f16_to_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)dY.data, (__half*)out.data, M, N);
  }

  return cuda_last_error_to_status();
}

KernelVariant make_reduce_sum_lastdim_f16_variant() {
  KernelVariant v{};
  v.name = "reduce_sum_keep_lastdim_f16_to_f16";
  v.priority = 115; // f16->f16을 f16->f32보다 우선시키고 싶으면 더 크게
  v.flags = 0;
  v.launch = reduce_sum_f16_launch;
  v.supported = reduce_sum_f16_supported;
  v.query_workspace = reduce_sum_workspace;
  return v;
}

// ============================================================================
// public API (core-free)
//   ⚠️ api.hpp도 반드시 이 시그니처로 맞춰야 함.
// ============================================================================
Status reduce_sum_lastdim_f32(const float* dY, float* dB, int M, int N, cudaStream_t stream) {
  if (!dY || !dB || M <= 0 || N <= 0) return Status::InvalidArgument;

  constexpr int kThreads = 256;
  dim3 block(kThreads, 1, 1);
  dim3 grid(N, 1, 1);

  reduce_sum_impl::reduce_sum_rows_f32_to_f32_kernel<<<grid, block, 0, stream>>>(dY, dB, M, N);
  return cuda_last_error_to_status();
}

Status reduce_sum_lastdim_f16_to_f32(const __half* dY, float* dB, int M, int N, cudaStream_t stream) {
  if (!dY || !dB || M <= 0 || N <= 0) return Status::InvalidArgument;

  constexpr int kThreads = 256;
  dim3 block(kThreads, 1, 1);

  if ((N & 1) == 0 && is_aligned_ptr(dY, 4) && is_aligned_ptr(dB, 4)) {
    dim3 grid(N / 2, 1, 1);
    reduce_sum_impl::reduce_sum_rows_f16x2_to_f32_kernel<<<grid, block, 0, stream>>>(
        (const __half2*)dY, dB, M, N / 2);
  } else {
    dim3 grid(N, 1, 1);
    reduce_sum_impl::reduce_sum_rows_f16_to_f32_kernel<<<grid, block, 0, stream>>>(
        dY, dB, M, N);
  }

  return cuda_last_error_to_status();
}

Status reduce_sum_lastdim_f16(const __half* dY, __half* dB, int M, int N, cudaStream_t stream) {
  if (!dY || !dB || M <= 0 || N <= 0) return Status::InvalidArgument;

  constexpr int kThreads = 256;
  dim3 block(kThreads, 1, 1);

  if ((N & 1) == 0 && is_aligned_ptr(dY, 4) && is_aligned_ptr(dB, 4)) {
    dim3 grid(N / 2, 1, 1);
    reduce_sum_impl::reduce_sum_rows_f16x2_to_f16_kernel<<<grid, block, 0, stream>>>(
        (const __half2*)dY, (__half2*)dB, M, N / 2);
  } else {
    dim3 grid(N, 1, 1);
    reduce_sum_impl::reduce_sum_rows_f16_to_f16_kernel<<<grid, block, 0, stream>>>(
        dY, dB, M, N);
  }

  return cuda_last_error_to_status();
}

} // namespace aicf::cuda
