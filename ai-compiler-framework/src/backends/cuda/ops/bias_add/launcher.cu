#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API
#include <aicf/backends/cuda/ops/bias_add/api.hpp>

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
// kernels (definitions live here)
//  - grid-stride 1D로 바꿔서 M,N이 커져도 grid.y 제한 안 걸리게.
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

// half2: last-dim(N)이 짝수이며, ptr들이 4B align일 때 사용
// 데이터는 [M, N] (row-major). half2로 보면 [M, N2] where N2=N/2.
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

// -------------------------
// public API implementation
// -------------------------
aicf::Status bias_add_f32(const float* Y,
                          const float* bias,
                          float* Out,
                          int M, int N,
                          aicf::Stream stream) {
  if (!Y || !bias || !Out || M <= 0 || N <= 0) {
    return aicf::Status::InvalidArgument;
  }

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  constexpr int kThreads = 256;
  // grid는 너무 크게 잡을 필요 없음. 통상 SM*몇 배 정도.
  // 여기선 단순하게 blocks = ceil(total/threads) 상한을 둠.
  const int64_t total = (int64_t)M * (int64_t)N;
  int blocks = (int)((total + kThreads - 1) / kThreads);
  if (blocks > 65535) blocks = 65535;

  bias_add_impl::bias_add_f32_kernel<<<blocks, kThreads, 0, s>>>(Y, bias, Out, M, N);
  return aicf::cuda::shim::cuda_last_error_to_status();
}

// -------------------------
// Attr helpers (local, minimal)
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

// dtype-generic check: dt check 함수 포인터로 중복 제거
static inline bool bias_add_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    int64_t axis_raw,
    bool (*is_YO_ok)(const TensorDesc&)) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& Y = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& O = outputs[0];

  if (!is_YO_ok(Y)) return false;
  if (!is_YO_ok(O)) return false;

  // bias: same dtype, contig 1D
  if (!(B.contiguous && B.rank() == 1 && B.dtype == Y.dtype)) return false;

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

// -------------------------
// Variant: F32
// -------------------------
static bool bias_add_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  int64_t axis = -1;
  (void)attr_get_i64(attr, "axis", &axis);
  return bias_add_check(inputs, num_inputs, outputs, num_outputs, axis,
                        &is_f32_contig_rank_ge2);
}

static aicf::Status bias_add_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  int64_t axis = -1;
  (void)attr_get_i64(attr, "axis", &axis);

  if (!bias_add_check(inputs, num_inputs, outputs, num_outputs, axis,
                      &is_f32_contig_rank_ge2)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& Y = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  int64_t M64 = 0, N64 = 0;
  if (!compute_MN_last_dim(Y, &M64, &N64)) return aicf::Status::InvalidArgument;

  const int M = (int)M64;
  const int N = (int)N64;

  constexpr int kThreads = 256;
  const int64_t total = (int64_t)M * (int64_t)N;
  int blocks = (int)((total + kThreads - 1) / kThreads);
  if (blocks > 65535) blocks = 65535;

  bias_add_impl::bias_add_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)Y.data, (const float*)B.data, (float*)O.data, M, N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_bias_add_f32_variant() {
  KernelVariant v{};
  v.name = "bias_add_f32";
  v.priority = 0;
  v.flags = 0;
  v.launch = bias_add_f32_launch;
  v.supported = bias_add_f32_supported;
  v.query_workspace = bias_add_workspace;
  return v;
}

// -------------------------
// Variant: F16 naive
// -------------------------
static bool bias_add_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  int64_t axis = -1;
  (void)attr_get_i64(attr, "axis", &axis);
  return bias_add_check(inputs, num_inputs, outputs, num_outputs, axis,
                        &is_f16_contig_rank_ge2);
}

static aicf::Status bias_add_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  int64_t axis = -1;
  (void)attr_get_i64(attr, "axis", &axis);

  if (!bias_add_check(inputs, num_inputs, outputs, num_outputs, axis,
                      &is_f16_contig_rank_ge2)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& Y = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  int64_t M64 = 0, N64 = 0;
  if (!compute_MN_last_dim(Y, &M64, &N64)) return aicf::Status::InvalidArgument;

  const int M = (int)M64;
  const int N = (int)N64;

  constexpr int kThreads = 256;
  const int64_t total = (int64_t)M * (int64_t)N;
  int blocks = (int)((total + kThreads - 1) / kThreads);
  if (blocks > 65535) blocks = 65535;

  bias_add_impl::bias_add_f16_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half*)Y.data, (const __half*)B.data, (__half*)O.data, M, N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_bias_add_f16_variant() {
  KernelVariant v{};
  v.name = "bias_add_f16_naive";
  v.priority = 0;
  v.flags = 0;
  v.launch = bias_add_f16_launch;
  v.supported = bias_add_f16_supported;
  v.query_workspace = bias_add_workspace;
  return v;
}

// -------------------------
// Variant: F16 half2 (vec2)
// -------------------------
static inline bool bias_add_f16_vec2_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    int64_t axis_raw) {

  if (!bias_add_check(inputs, num_inputs, outputs, num_outputs, axis_raw,
                      &is_f16_contig_rank_ge2)) {
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
  if (!aicf::cuda::shim::is_aligned_data(Y, kAlign)) return false;
  if (!aicf::cuda::shim::is_aligned_data(B, kAlign)) return false;
  if (!aicf::cuda::shim::is_aligned_data(O, kAlign)) return false;

  return true;
}

static bool bias_add_f16_vec2_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  int64_t axis = -1;
  (void)attr_get_i64(attr, "axis", &axis);
  return bias_add_f16_vec2_check(inputs, num_inputs, outputs, num_outputs, axis);
}

static aicf::Status bias_add_f16_vec2_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  int64_t axis = -1;
  (void)attr_get_i64(attr, "axis", &axis);

  if (!bias_add_f16_vec2_check(inputs, num_inputs, outputs, num_outputs, axis)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& Y = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  int64_t M64 = 0, N64 = 0;
  if (!compute_MN_last_dim(Y, &M64, &N64)) return aicf::Status::InvalidArgument;

  const int M = (int)M64;
  const int N = (int)N64;
  const int N2 = N / 2;

  constexpr int kThreads = 256;
  const int64_t total2 = (int64_t)M * (int64_t)N2;
  int blocks = (int)((total2 + kThreads - 1) / kThreads);
  if (blocks > 65535) blocks = 65535;

  bias_add_impl::bias_add_f16x2_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half2*)Y.data, (const __half2*)B.data, (__half2*)O.data, M, N2);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_bias_add_f16_vec2_variant() {
  KernelVariant v{};
  v.name = "bias_add_f16_vec2_half2";
  v.priority = 10;
  v.flags = 0;
  v.launch = bias_add_f16_vec2_launch;
  v.supported = bias_add_f16_vec2_supported;
  v.query_workspace = bias_add_workspace;
  return v;
}

} // namespace aicf::cuda
