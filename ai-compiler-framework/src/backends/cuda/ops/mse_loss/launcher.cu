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
// AttrBlob schema: MSELoss
// schema_id: 'MSEL' (0x4C45534D little-endian)
// payload:
//   int32 reduction  (0=mean, 1=sum)
// schema_id==0 allowed -> default mean
// -------------------------
static constexpr uint32_t kSchema_MSEL = 0x4C45534Du;

static inline int32_t read_i32_le(const uint8_t* p) {
  int32_t v;
  std::memcpy(&v, p, sizeof(int32_t));
  return v;
}

static inline AttrBlob as_attr_blob(const void* attr) {
  if (!attr) return AttrBlob{0, 0, nullptr};
  return *static_cast<const AttrBlob*>(attr);
}

static inline int get_reduction_from_attr(const AttrBlob& a) {
  // default mean
  if (a.schema_id == 0) return 0;
  if (a.schema_id != kSchema_MSEL) return 0;
  if (a.bytes < 4 || !a.data) return 0;
  int32_t r = read_i32_le((const uint8_t*)a.data);
  return (r == 1) ? 1 : 0;
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

static inline bool is_f32_contig_scalar1(const TensorDesc& T) {
  if (T.dtype != DType::kF32) return false;
  if (!T.contiguous) return false;
  int64_t numel = 0;
  if (!compute_numel(T, &numel)) return false;
  return (numel == 1);
}

static inline int choose_blocks_1d(int64_t numel, int threads) {
  int64_t blocks64 = (numel + threads - 1) / threads;
  if (blocks64 < 1) blocks64 = 1;
  const int64_t kMaxBlocks = 65535;
  if (blocks64 > kMaxBlocks) blocks64 = kMaxBlocks;
  return (int)blocks64;
}

static size_t mse_loss_workspace(const TensorDesc*, int, const void*) { return 0; }

// ============================================================================
// kernels
// ============================================================================
namespace mse_loss_impl {

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

__global__ void mse_loss_sum_f32_kernel(const float* __restrict__ pred,
                                       const float* __restrict__ target,
                                       float* __restrict__ out,
                                       int64_t numel) {
  float acc = 0.0f;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < numel;
       i += (int64_t)gridDim.x * blockDim.x) {
    float d = pred[i] - target[i];
    acc += d * d;
  }
  float sum = block_reduce_sum(acc);
  if (threadIdx.x == 0) atomicAdd(out, sum);
}

__global__ void mse_loss_sum_f16_kernel(const __half* __restrict__ pred,
                                       const __half* __restrict__ target,
                                       float* __restrict__ out,
                                       int64_t numel) {
  float acc = 0.0f;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < numel;
       i += (int64_t)gridDim.x * blockDim.x) {
    float p = __half2float(pred[i]);
    float t = __half2float(target[i]);
    float d = p - t;
    acc += d * d;
  }
  float sum = block_reduce_sum(acc);
  if (threadIdx.x == 0) atomicAdd(out, sum);
}

__global__ void scale_f32_scalar_kernel(float* __restrict__ out, float scale) {
  out[0] *= scale;
}

} // namespace mse_loss_impl

// ============================================================================
// Variant: MSELoss f32 -> f32 scalar
// inputs=(pred, target), outputs=(loss[1])
// forbid alias: out must not alias inputs
// ============================================================================
static inline bool mse_loss_check_f32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& P = inputs[0];
  const TensorDesc& T = inputs[1];
  const TensorDesc& O = outputs[0];

  if (!is_f32_contig_anyrank(P) || !is_f32_contig_anyrank(T)) return false;
  if (!same_shape(P, T)) return false;
  if (!is_f32_contig_scalar1(O)) return false;

  int64_t numel = 0;
  if (!compute_numel(P, &numel)) return false;
  return (numel > 0);
}

static bool mse_loss_supported_f32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return mse_loss_check_f32(inputs, num_inputs, outputs, num_outputs);
}

static Status mse_loss_launch_f32(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!mse_loss_check_f32(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& P = inputs[0];
  const TensorDesc& T = inputs[1];
  TensorDesc& O = outputs[0];

  if (O.data == P.data || O.data == T.data) return Status::InvalidArgument;

  int64_t numel = 0;
  (void)compute_numel(P, &numel);

  const AttrBlob a = as_attr_blob(attr);
  const int reduction = get_reduction_from_attr(a); // 0 mean, 1 sum

  // out = 0
  cudaError_t e = cudaMemsetAsync(O.data, 0, sizeof(float), stream);
  if (e != cudaSuccess) return cuda_to_status(e);

  constexpr int kThreads = 256;
  const int blocks = choose_blocks_1d(numel, kThreads);

  cudaGetLastError(); // clear
  mse_loss_impl::mse_loss_sum_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)P.data, (const float*)T.data, (float*)O.data, numel);
  Status st = cuda_last_status();
  if (!ok(st)) return st;

  if (reduction == 0) { // mean
    const float scale = 1.0f / (float)numel;
    mse_loss_impl::scale_f32_scalar_kernel<<<1, 1, 0, stream>>>((float*)O.data, scale);
    st = cuda_last_status();
  }

  return st;
}

KernelVariant make_mse_loss_f32_variant() {
  KernelVariant v{};
  v.name = "mse_loss_f32";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = 0; // allow default
  v.launch = mse_loss_launch_f32;
  v.supported = mse_loss_supported_f32;
  v.query_workspace = mse_loss_workspace;
  return v;
}

// ============================================================================
// Variant: MSELoss f16 -> f32 scalar
// ============================================================================
static inline bool mse_loss_check_f16(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& P = inputs[0];
  const TensorDesc& T = inputs[1];
  const TensorDesc& O = outputs[0];

  if (!is_f16_contig_anyrank(P) || !is_f16_contig_anyrank(T)) return false;
  if (!same_shape(P, T)) return false;
  if (!is_f32_contig_scalar1(O)) return false;

  int64_t numel = 0;
  if (!compute_numel(P, &numel)) return false;
  return (numel > 0);
}

static bool mse_loss_supported_f16(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return mse_loss_check_f16(inputs, num_inputs, outputs, num_outputs);
}

static Status mse_loss_launch_f16(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!mse_loss_check_f16(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& P = inputs[0];
  const TensorDesc& T = inputs[1];
  TensorDesc& O = outputs[0];

  if (O.data == P.data || O.data == T.data) return Status::InvalidArgument;

  int64_t numel = 0;
  (void)compute_numel(P, &numel);

  const AttrBlob a = as_attr_blob(attr);
  const int reduction = get_reduction_from_attr(a);

  cudaError_t e = cudaMemsetAsync(O.data, 0, sizeof(float), stream);
  if (e != cudaSuccess) return cuda_to_status(e);

  constexpr int kThreads = 256;
  const int blocks = choose_blocks_1d(numel, kThreads);

  cudaGetLastError(); // clear
  mse_loss_impl::mse_loss_sum_f16_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half*)P.data, (const __half*)T.data, (float*)O.data, numel);
  Status st = cuda_last_status();
  if (!ok(st)) return st;

  if (reduction == 0) {
    const float scale = 1.0f / (float)numel;
    mse_loss_impl::scale_f32_scalar_kernel<<<1, 1, 0, stream>>>((float*)O.data, scale);
    st = cuda_last_status();
  }

  return st;
}

KernelVariant make_mse_loss_f16_variant() {
  KernelVariant v{};
  v.name = "mse_loss_f16_to_f32";
  v.priority = 10;
  v.flags = 0;
  v.expected_attr_schema_id = 0;
  v.launch = mse_loss_launch_f16;
  v.supported = mse_loss_supported_f16;
  v.query_workspace = mse_loss_workspace;
  return v;
}

} // namespace aicf::cuda
