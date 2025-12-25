#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API
#include <aicf/backends/cuda/ops/relu_bwd/api.hpp>

// registry glue
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>

// common shim
#include "aicf/backends/cuda/ops/_common/shim/launch.hpp"
#include "aicf/backends/cuda/ops/_common/shim/status.hpp"
#include "aicf/backends/cuda/ops/_common/shim/validate.hpp"

#include "kernels.cuh"

namespace aicf::cuda {

// -------------------------
// kernels
// -------------------------
namespace relu_bwd_impl {

__global__ void relu_bwd_f32_kernel(const float* __restrict__ Y,
                                   const float* __restrict__ dOut,
                                   float* __restrict__ dY,
                                   int64_t numel) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (i >= numel) return;
  const float y = Y[i];
  dY[i] = (y > 0.0f) ? dOut[i] : 0.0f;
}

__global__ void relu_bwd_f16_kernel(const __half* __restrict__ Y,
                                   const __half* __restrict__ dOut,
                                   __half* __restrict__ dY,
                                   int64_t numel) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (i >= numel) return;

  const __half y = Y[i];
  // y > 0 ? dOut : 0
  dY[i] = __hgt(y, __float2half(0.0f)) ? dOut[i] : __float2half(0.0f);
}

__global__ void relu_bwd_f16x2_kernel(const __half2* __restrict__ Y,
                                     const __half2* __restrict__ dOut,
                                     __half2* __restrict__ dY,
                                     int64_t numel2) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (i >= numel2) return;

  const __half2 y = Y[i];
  const __half2 g = dOut[i];
  const __half2 z = __float2half2_rn(0.0f);

  // mask: (y > 0) ? 1 : 0 (lane-wise)
  // CUDA half2 비교는 bool2 형태로 직접 다루기 애매하니,
  // relu_bwd는 "y<=0이면 0"이므로 아래처럼 max(y,0) 기반으로 처리:
  //   (y>0) => relu(y)=y, (y<=0)=>relu(y)=0
  //   g * (y>0) 를 만들기 위해 (relu(y) == 0)면 0, 아니면 g를 선택
  // 여기서는 branch-free로:
  //   out = (y > 0) ? g : 0 를 각 lane에 대해 조건 선택
  //
  // 가장 단순하고 안정적인 방식: 두 lane을 스칼라로 분해 (성능 손해 작음, 그래도 load/store는 vec2)
  // 성능을 더 밀고 싶으면 bitmask 선택을 별도 최적화하면 됨.
  const __half y0 = __low2half(y);
  const __half y1 = __high2half(y);
  const __half g0 = __low2half(g);
  const __half g1 = __high2half(g);

  const __half o0 = __hgt(y0, __float2half(0.0f)) ? g0 : __float2half(0.0f);
  const __half o1 = __hgt(y1, __float2half(0.0f)) ? g1 : __float2half(0.0f);

  dY[i] = __halves2half2(o0, o1);
}

} // namespace relu_bwd_impl

// -------------------------
// public API implementation
// -------------------------
aicf::Status relu_bwd_f32(const float* Y,
                          const float* dOut,
                          float* dY,
                          int64_t numel,
                          aicf::Stream stream) {
  if (!Y || !dOut || !dY || numel <= 0) {
    return aicf::Status::InvalidArgument;
  }

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  constexpr int kThreads = 256;
  int blocks = (int)((numel + kThreads - 1) / kThreads);
  if (blocks > 65535) blocks = 65535;

  relu_bwd_impl::relu_bwd_f32_kernel<<<blocks, kThreads, 0, s>>>(Y, dOut, dY, numel);
  return aicf::cuda::shim::cuda_last_error_to_status();
}

// (옵션) public API를 f16도 열고 싶으면 api.hpp에 추가.
// 지금은 패턴만 맞춰서 void* 버전으로 제안:
aicf::Status relu_bwd_f16(const void* Y,
                          const void* dOut,
                          void* dY,
                          int64_t numel,
                          aicf::Stream stream) {
  if (!Y || !dOut || !dY || numel <= 0) {
    return aicf::Status::InvalidArgument;
  }

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  constexpr int kThreads = 256;

  const bool even = ((numel & 1) == 0);
  const bool y_aligned = (((uintptr_t)Y & 0x3u) == 0);
  const bool g_aligned = (((uintptr_t)dOut & 0x3u) == 0);
  const bool o_aligned = (((uintptr_t)dY & 0x3u) == 0);

  if (even && y_aligned && g_aligned && o_aligned) {
    const int64_t numel2 = numel / 2;
    int blocks = (int)((numel2 + kThreads - 1) / kThreads);
    if (blocks > 65535) blocks = 65535;

    relu_bwd_impl::relu_bwd_f16x2_kernel<<<blocks, kThreads, 0, s>>>(
        (const __half2*)Y, (const __half2*)dOut, (__half2*)dY, numel2);
  } else {
    int blocks = (int)((numel + kThreads - 1) / kThreads);
    if (blocks > 65535) blocks = 65535;

    relu_bwd_impl::relu_bwd_f16_kernel<<<blocks, kThreads, 0, s>>>(
        (const __half*)Y, (const __half*)dOut, (__half*)dY, numel);
  }

  return aicf::cuda::shim::cuda_last_error_to_status();
}

// -------------------------
// helpers
// -------------------------
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

// -------------------------
// Contract:
// inputs[0] = Y
// inputs[1] = dOut
// outputs[0]= dY
// same shape, contig, dtype per variant
// -------------------------
static inline bool relu_bwd_check_dt(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    bool (*is_ok)(const TensorDesc&)) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& Y = inputs[0];
  const TensorDesc& dOut = inputs[1];
  const TensorDesc& dY = outputs[0];

  if (!is_ok(Y) || !is_ok(dOut) || !is_ok(dY)) return false;
  if (Y.rank() < 1) return false;
  if (!same_shape(Y, dOut)) return false;
  if (!same_shape(Y, dY)) return false;

  int64_t numel = 0;
  if (!compute_numel(Y, &numel)) return false;
  return true;
}

static size_t relu_bwd_workspace(const TensorDesc*, int, const void*) { return 0; }

// ---- F32 variant ----
static bool relu_bwd_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return relu_bwd_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f32_contig);
}

static aicf::Status relu_bwd_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!relu_bwd_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f32_contig)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& Y = inputs[0];
  const TensorDesc& dOut = inputs[1];
  TensorDesc& dY = outputs[0];

  int64_t numel = 0;
  if (!compute_numel(Y, &numel)) return aicf::Status::InvalidArgument;

  constexpr int kThreads = 256;
  int blocks = (int)((numel + kThreads - 1) / kThreads);
  if (blocks > 65535) blocks = 65535;

  relu_bwd_impl::relu_bwd_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)Y.data, (const float*)dOut.data, (float*)dY.data, numel);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_relu_bwd_f32_variant() {
  KernelVariant v{};
  v.name = "relu_bwd_f32";
  v.priority = 0;
  v.flags = 0;
  v.launch = relu_bwd_f32_launch;
  v.supported = relu_bwd_f32_supported;
  v.query_workspace = relu_bwd_workspace;
  return v;
}

// ---- F16 naive variant ----
static bool relu_bwd_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return relu_bwd_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f16_contig);
}

static aicf::Status relu_bwd_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!relu_bwd_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f16_contig)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& Y = inputs[0];
  const TensorDesc& dOut = inputs[1];
  TensorDesc& dY = outputs[0];

  int64_t numel = 0;
  if (!compute_numel(Y, &numel)) return aicf::Status::InvalidArgument;

  constexpr int kThreads = 256;
  int blocks = (int)((numel + kThreads - 1) / kThreads);
  if (blocks > 65535) blocks = 65535;

  relu_bwd_impl::relu_bwd_f16_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half*)Y.data, (const __half*)dOut.data, (__half*)dY.data, numel);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_relu_bwd_f16_variant() {
  KernelVariant v{};
  v.name = "relu_bwd_f16_naive";
  v.priority = 0;
  v.flags = 0;
  v.launch = relu_bwd_f16_launch;
  v.supported = relu_bwd_f16_supported;
  v.query_workspace = relu_bwd_workspace;
  return v;
}

// ---- F16 vec2 (half2) variant ----
static inline bool relu_bwd_f16_vec2_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!relu_bwd_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f16_contig)) {
    return false;
  }

  const TensorDesc& Y = inputs[0];
  const TensorDesc& dOut = inputs[1];
  const TensorDesc& dY = outputs[0];

  int64_t numel = 0;
  if (!compute_numel(Y, &numel)) return false;
  if ((numel & 1) != 0) return false;

  constexpr size_t kAlign = 4;
  if (!aicf::cuda::shim::is_aligned_data(Y, kAlign)) return false;
  if (!aicf::cuda::shim::is_aligned_data(dOut, kAlign)) return false;
  if (!aicf::cuda::shim::is_aligned_data(dY, kAlign)) return false;

  return true;
}

static bool relu_bwd_f16_vec2_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return relu_bwd_f16_vec2_check(inputs, num_inputs, outputs, num_outputs);
}

static aicf::Status relu_bwd_f16_vec2_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!relu_bwd_f16_vec2_check(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& Y = inputs[0];
  const TensorDesc& dOut = inputs[1];
  TensorDesc& dY = outputs[0];

  int64_t numel = 0;
  if (!compute_numel(Y, &numel)) return aicf::Status::InvalidArgument;

  const int64_t numel2 = numel / 2;

  constexpr int kThreads = 256;
  int blocks = (int)((numel2 + kThreads - 1) / kThreads);
  if (blocks > 65535) blocks = 65535;

  relu_bwd_impl::relu_bwd_f16x2_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half2*)Y.data, (const __half2*)dOut.data, (__half2*)dY.data, numel2);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_relu_bwd_f16_vec2_variant() {
  KernelVariant v{};
  v.name = "relu_bwd_f16_vec2_half2";
  v.priority = 10;
  v.flags = 0;
  v.launch = relu_bwd_f16_vec2_launch;
  v.supported = relu_bwd_f16_vec2_supported;
  v.query_workspace = relu_bwd_workspace;
  return v;
}

} // namespace aicf::cuda
