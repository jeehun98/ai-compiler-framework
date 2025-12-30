// ============================================================================
// src/backends/cuda/ops/layernorm/launcher.cu
// - KEEP "kernel definitions inside launcher.cu" structure
// - 2D only: x[M,N], normalize over last-dim (N)
// - Optional affine: gamma/beta may be omitted or null
// - mean/rstd outputs: float[M] (for backward reuse)
//
// [DEBUG] Adds stderr prints at launch entry to validate pointers/shapes/eps.
// [FIX]   block_sum_f32 now broadcasts block-wide sum correctly.
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <string_view>
#include <cmath>
#include <cstdio>   // [DEBUG]

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

#include <aicf/backends/cuda/ops/layernorm/api.hpp>

#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_pack.hpp>

#include "aicf/backends/cuda/ops/_common/shim/launch.hpp"
#include "aicf/backends/cuda/ops/_common/shim/status.hpp"
#include "aicf/backends/cuda/ops/_common/shim/validate.hpp"

#include "kernels.cuh"

namespace aicf::cuda {

// ============================================================================
// Attr helpers (same pattern as gemm)
// ============================================================================

static inline bool attr_get_bool(const void* attr, const char* key, bool default_val) {
  if (!attr) return default_val;
  const auto* pack = static_cast<const aicf::cuda::AttrPack*>(attr);
  if (!pack->items || pack->size <= 0) return default_val;

  const std::string_view k(key);
  for (int i = 0; i < pack->size; ++i) {
    const auto& kv = pack->items[i];
    if (!kv.key) continue;
    if (std::string_view(kv.key) == k) {
      if (kv.val.tag == aicf::cuda::AttrTag::kBool) return kv.val.b32 != 0;
      return default_val;
    }
  }
  return default_val;
}

static inline float attr_get_f32(const void* attr, const char* key, float default_val) {
  if (!attr) return default_val;
  const auto* pack = static_cast<const aicf::cuda::AttrPack*>(attr);
  if (!pack->items || pack->size <= 0) return default_val;

  const std::string_view k(key);
  for (int i = 0; i < pack->size; ++i) {
    const auto& kv = pack->items[i];
    if (!kv.key) continue;
    if (std::string_view(kv.key) == k) {
      if (kv.val.tag == aicf::cuda::AttrTag::kF32) return kv.val.f32;
      return default_val;
    }
  }
  return default_val;
}

// ============================================================================
// Tensor helpers
// ============================================================================

static inline bool is_2d(const TensorDesc& T) { return T.rank() == 2; }
static inline bool stride_valid_2d(const TensorDesc& T) {
  if (!is_2d(T)) return false;
  return (T.stride[0] > 0 && T.stride[1] > 0);
}
static inline bool is_contig_rowmajor_2d(const TensorDesc& T) {
  if (!is_2d(T)) return false;
  if (!stride_valid_2d(T)) return false;
  return (T.stride[1] == 1) && (T.stride[0] == T.shape[1]);
}

static inline bool is_f32_2d(const TensorDesc& T) { return (T.dtype == DType::kF32) && is_2d(T); }
static inline bool is_f16_2d(const TensorDesc& T) { return (T.dtype == DType::kF16) && is_2d(T); }

// ============================================================================
// kernels (definitions)  -- keep here
// ============================================================================

namespace layernorm_impl {

__device__ __forceinline__ float warp_sum_f32(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

// [FIX] block-wide sum broadcast
__device__ __forceinline__ float block_sum_f32(float v) {
  __shared__ float smem[32];  // up to 1024 threads => 32 warps
  __shared__ float total;     // broadcast to whole block
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  v = warp_sum_f32(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();

  if (warp == 0) {
    float x = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
    x = warp_sum_f32(x);
    if (lane == 0) total = x;
  }
  __syncthreads();
  return total;
}

__global__ void layernorm_fwd_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int M, int N,
    float eps) {

  const int row = (int)blockIdx.x;
  if (row >= M) return;

  const float* xr = x + (int64_t)row * (int64_t)N;
  float* yr = y + (int64_t)row * (int64_t)N;

  float s1 = 0.0f;
  float s2 = 0.0f;
  for (int c = threadIdx.x; c < N; c += blockDim.x) {
    const float v = xr[c];
    s1 += v;
    s2 += v * v;
  }
  s1 = block_sum_f32(s1);
  s2 = block_sum_f32(s2);

  const float invN = 1.0f / (float)N;
  const float mu = s1 * invN;
  const float var = fmaxf(s2 * invN - mu * mu, 0.0f);
  const float rs = rsqrtf(var + eps);

  if (threadIdx.x == 0) {
    mean[row] = mu;
    rstd[row] = rs;
  }

  const bool has_affine = (gamma != nullptr) && (beta != nullptr);

  for (int c = threadIdx.x; c < N; c += blockDim.x) {
    const float v = xr[c];
    float nh = (v - mu) * rs;
    if (has_affine) nh = nh * gamma[c] + beta[c];
    yr[c] = nh;
  }
}

__global__ void layernorm_fwd_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    __half* __restrict__ y,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int M, int N,
    float eps) {

  const int row = (int)blockIdx.x;
  if (row >= M) return;

  const __half* xr = x + (int64_t)row * (int64_t)N;
  __half* yr = y + (int64_t)row * (int64_t)N;

  float s1 = 0.0f;
  float s2 = 0.0f;
  for (int c = threadIdx.x; c < N; c += blockDim.x) {
    const float v = __half2float(xr[c]);
    s1 += v;
    s2 += v * v;
  }
  s1 = block_sum_f32(s1);
  s2 = block_sum_f32(s2);

  const float invN = 1.0f / (float)N;
  const float mu = s1 * invN;
  const float var = fmaxf(s2 * invN - mu * mu, 0.0f);
  const float rs = rsqrtf(var + eps);

  if (threadIdx.x == 0) {
    mean[row] = mu;
    rstd[row] = rs;
  }

  const bool has_affine = (gamma != nullptr) && (beta != nullptr);

  for (int c = threadIdx.x; c < N; c += blockDim.x) {
    const float v = __half2float(xr[c]);
    float nh = (v - mu) * rs;
    if (has_affine) {
      nh = nh * __half2float(gamma[c]) + __half2float(beta[c]);
    }
    yr[c] = __float2half_rn(nh);
  }
}

} // namespace layernorm_impl

// ============================================================================
// Registry variant: LayerNorm forward (f32)
//   inputs: 1 (x) or 3 (x,gamma,beta)
//   outputs: 3 (y, mean, rstd)
//   attr: eps (f32, default=1e-5)
// ============================================================================

static bool ln_fwd_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  if (!inputs || !outputs) return false;
  if (!(num_inputs == 1 || num_inputs == 3)) return false;
  if (num_outputs != 3) return false;

  const TensorDesc& X = inputs[0];
  const TensorDesc& Y = outputs[0];
  const TensorDesc& Mean = outputs[1];
  const TensorDesc& Rstd = outputs[2];

  if (!is_f32_2d(X) || !is_f32_2d(Y)) return false;
  if (!is_contig_rowmajor_2d(X) || !is_contig_rowmajor_2d(Y)) return false;

  if (!(Mean.rank() == 1 && Mean.dtype == DType::kF32)) return false;
  if (!(Rstd.rank() == 1 && Rstd.dtype == DType::kF32)) return false;
  if (!(Mean.shape[0] == X.shape[0] && Rstd.shape[0] == X.shape[0])) return false;

  if (!(Y.shape[0] == X.shape[0] && Y.shape[1] == X.shape[1])) return false;

  if (num_inputs == 3) {
    const TensorDesc& G = inputs[1];
    const TensorDesc& B = inputs[2];
    if (!(G.rank() == 1 && B.rank() == 1)) return false;
    if (!(G.dtype == DType::kF32 && B.dtype == DType::kF32)) return false;
    if (!(G.shape[0] == X.shape[1] && B.shape[0] == X.shape[1])) return false;
  }

  return true;
}

static size_t ln_fwd_f32_workspace(const TensorDesc*, int, const void*) { return 0; }

static aicf::Status ln_fwd_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!ln_fwd_f32_supported(inputs, num_inputs, outputs, num_outputs, attr)) {
    return aicf::Status::InvalidArgument;
  }

  const float eps = attr_get_f32(attr, "eps", 1e-5f);

  const TensorDesc& X = inputs[0];
  TensorDesc& Y = outputs[0];
  TensorDesc& Mean = outputs[1];
  TensorDesc& Rstd = outputs[2];

  const float* x = (const float*)X.data;
  float* y = (float*)Y.data;
  float* mean = (float*)Mean.data;
  float* rstd = (float*)Rstd.data;

  const float* gamma = nullptr;
  const float* beta  = nullptr;
  if (num_inputs == 3) {
    gamma = (const float*)inputs[1].data;
    beta  = (const float*)inputs[2].data;
  }

  const int M = (int)X.shape[0];
  const int N = (int)X.shape[1];

  int threads = 256;
  if (N < threads) {
    threads = 1;
    while (threads < N) threads <<= 1;
    if (threads > 256) threads = 256;
  }

  dim3 block(threads, 1, 1);
  dim3 grid(M, 1, 1);

  // [DEBUG]
  std::fprintf(stderr,
    "[LN f32] num_inputs=%d num_outputs=%d "
    "X=%p G=%p B=%p | Y=%p Mean=%p Rstd=%p | M=%d N=%d | eps=%g\n",
    num_inputs, num_outputs,
    (const void*)x,
    (const void*)gamma,
    (const void*)beta,
    (const void*)y,
    (const void*)mean,
    (const void*)rstd,
    M, N,
    (double)eps
  );

  layernorm_impl::layernorm_fwd_f32_kernel<<<grid, block, 0, stream>>>(
      x, gamma, beta, y, mean, rstd, M, N, eps);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_layernorm_fwd_f32_variant() {
  KernelVariant v{};
  v.name = "layernorm_fwd_f32_contig2d";
  v.priority = 0;
  v.flags = 0;
  v.launch = ln_fwd_f32_launch;
  v.supported = ln_fwd_f32_supported;
  v.query_workspace = ln_fwd_f32_workspace;
  return v;
}

// ============================================================================
// Registry variant: LayerNorm forward (f16 x/y + f16 gamma/beta, mean/rstd f32)
// ============================================================================

static bool ln_fwd_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  if (!inputs || !outputs) return false;
  if (!(num_inputs == 1 || num_inputs == 3)) return false;
  if (num_outputs != 3) return false;

  const TensorDesc& X = inputs[0];
  const TensorDesc& Y = outputs[0];
  const TensorDesc& Mean = outputs[1];
  const TensorDesc& Rstd = outputs[2];

  if (!is_f16_2d(X) || !is_f16_2d(Y)) return false;
  if (!is_contig_rowmajor_2d(X) || !is_contig_rowmajor_2d(Y)) return false;

  if (!(Mean.rank() == 1 && Mean.dtype == DType::kF32)) return false;
  if (!(Rstd.rank() == 1 && Rstd.dtype == DType::kF32)) return false;
  if (!(Mean.shape[0] == X.shape[0] && Rstd.shape[0] == X.shape[0])) return false;

  if (!(Y.shape[0] == X.shape[0] && Y.shape[1] == X.shape[1])) return false;

  if (num_inputs == 3) {
    const TensorDesc& G = inputs[1];
    const TensorDesc& B = inputs[2];
    if (!(G.rank() == 1 && B.rank() == 1)) return false;
    if (!(G.dtype == DType::kF16 && B.dtype == DType::kF16)) return false;
    if (!(G.shape[0] == X.shape[1] && B.shape[0] == X.shape[1])) return false;
  }

  return true;
}

static size_t ln_fwd_f16_workspace(const TensorDesc*, int, const void*) { return 0; }

static aicf::Status ln_fwd_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!ln_fwd_f16_supported(inputs, num_inputs, outputs, num_outputs, attr)) {
    return aicf::Status::InvalidArgument;
  }

  const float eps = attr_get_f32(attr, "eps", 1e-5f);

  const TensorDesc& X = inputs[0];
  TensorDesc& Y = outputs[0];
  TensorDesc& Mean = outputs[1];
  TensorDesc& Rstd = outputs[2];

  const __half* x = (const __half*)X.data;
  __half* y = (__half*)Y.data;
  float* mean = (float*)Mean.data;
  float* rstd = (float*)Rstd.data;

  const __half* gamma = nullptr;
  const __half* beta  = nullptr;
  if (num_inputs == 3) {
    gamma = (const __half*)inputs[1].data;
    beta  = (const __half*)inputs[2].data;
  }

  const int M = (int)X.shape[0];
  const int N = (int)X.shape[1];

  int threads = 256;
  if (N < threads) {
    threads = 1;
    while (threads < N) threads <<= 1;
    if (threads > 256) threads = 256;
  }

  dim3 block(threads, 1, 1);
  dim3 grid(M, 1, 1);

  // [DEBUG]
  std::fprintf(stderr,
    "[LN f16] num_inputs=%d num_outputs=%d "
    "X=%p G=%p B=%p | Y=%p Mean=%p Rstd=%p | M=%d N=%d | eps=%g | threads=%d\n",
    num_inputs, num_outputs,
    (const void*)x,
    (const void*)gamma,
    (const void*)beta,
    (const void*)y,
    (const void*)mean,
    (const void*)rstd,
    M, N,
    (double)eps,
    threads
  );

  layernorm_impl::layernorm_fwd_f16_kernel<<<grid, block, 0, stream>>>(
      x, gamma, beta, y, mean, rstd, M, N, eps);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_layernorm_fwd_f16_variant() {
  KernelVariant v{};
  v.name = "layernorm_fwd_f16_contig2d";
  v.priority = 10;
  v.flags = 0;
  v.launch = ln_fwd_f16_launch;
  v.supported = ln_fwd_f16_supported;
  v.query_workspace = ln_fwd_f16_workspace;
  return v;
}

// ============================================================================
// C++ API entrypoints (raw pointer path)  -- matches include/api.hpp
// ============================================================================

static inline cudaStream_t unwrap(Stream s) {
  return s.handle ? (cudaStream_t)s.handle : (cudaStream_t)0;
}

Status layernorm_fwd_f32(const float* x,
                         const float* gamma,
                         const float* beta,
                         float* y,
                         float* mean,
                         float* rstd,
                         int M, int N,
                         float eps,
                         Stream stream) {

  if (!x || !y || !mean || !rstd) return Status::InvalidArgument;
  if (M <= 0 || N <= 0) return Status::InvalidArgument;

  cudaStream_t st = unwrap(stream);

  int threads = 256;
  if (N < threads) {
    threads = 1;
    while (threads < N) threads <<= 1;
    if (threads > 256) threads = 256;
  }
  dim3 block(threads, 1, 1);
  dim3 grid(M, 1, 1);

  // [DEBUG]
  std::fprintf(stderr,
    "[LN api f32] X=%p G=%p B=%p | Y=%p Mean=%p Rstd=%p | M=%d N=%d eps=%g threads=%d\n",
    (const void*)x, (const void*)gamma, (const void*)beta,
    (const void*)y, (const void*)mean, (const void*)rstd,
    M, N, (double)eps, threads
  );

  layernorm_impl::layernorm_fwd_f32_kernel<<<grid, block, 0, st>>>(
      x, gamma, beta, y, mean, rstd, M, N, eps);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

Status layernorm_fwd_f16(const void* x,
                         const void* gamma,
                         const void* beta,
                         void* y,
                         float* mean,
                         float* rstd,
                         int M, int N,
                         float eps,
                         Stream stream) {

  if (!x || !y || !mean || !rstd) return Status::InvalidArgument;
  if (M <= 0 || N <= 0) return Status::InvalidArgument;

  cudaStream_t st = unwrap(stream);

  int threads = 256;
  if (N < threads) {
    threads = 1;
    while (threads < N) threads <<= 1;
    if (threads > 256) threads = 256;
  }
  dim3 block(threads, 1, 1);
  dim3 grid(M, 1, 1);

  // [DEBUG]
  std::fprintf(stderr,
    "[LN api f16] X=%p G=%p B=%p | Y=%p Mean=%p Rstd=%p | M=%d N=%d eps=%g threads=%d\n",
    x, gamma, beta, y, (const void*)mean, (const void*)rstd,
    M, N, (double)eps, threads
  );

  layernorm_impl::layernorm_fwd_f16_kernel<<<grid, block, 0, st>>>(
      (const __half*)x,
      (const __half*)gamma,
      (const __half*)beta,
      (__half*)y,
      mean,
      rstd,
      M, N, eps);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

// Backward stubs
Status layernorm_bwd_f32(float*, float*, float*, const float*, const float*,
                         const float*, const float*, const float*, int, int, Stream) {
  return Status::NotImplemented;
}
Status layernorm_bwd_f16(void*, void*, void*, const void*, const void*,
                         const float*, const float*, const void*, int, int, Stream) {
  return Status::NotImplemented;
}

} // namespace aicf::cuda
