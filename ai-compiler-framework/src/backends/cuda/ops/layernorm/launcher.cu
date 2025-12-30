// ============================================================================
// src/backends/cuda/ops/layernorm/launcher.cu
// - KEEP "kernel definitions inside launcher.cu" structure
// - 2D only: x[M,N], normalize over last-dim (N)
// - Optional affine forward: gamma/beta may be omitted or null
// - mean/rstd outputs: float[M]
// - Backward:
//    affine=True : outputs dx + dgamma(f32) + dbeta(f32)
//    affine=False: output  dx only
//
// Notes:
// - This is correctness-first. dgamma/dbeta are computed by a simple per-column reduce.
// - Debug prints included (stderr).
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <string_view>
#include <cmath>
#include <cstdio>

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
// Attr helpers
// ============================================================================

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

// block-wide sum broadcast (fixed)
__device__ __forceinline__ float block_sum_f32(float v) {
  __shared__ float smem[32];
  __shared__ float total;
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

// -------------------- forward --------------------

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

  float s1 = 0.0f, s2 = 0.0f;
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
    float xhat = (v - mu) * rs;
    if (has_affine) xhat = xhat * gamma[c] + beta[c];
    yr[c] = xhat;
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

  float s1 = 0.0f, s2 = 0.0f;
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
    float xhat = (v - mu) * rs;
    if (has_affine) {
      xhat = xhat * __half2float(gamma[c]) + __half2float(beta[c]);
    }
    yr[c] = __float2half_rn(xhat);
  }
}

// -------------------- backward: dx --------------------
// Formula (per-row):
//   dy_hat = dy * gamma  (gamma=1 if affine off)
//   dx = (1/N) * rstd * ( N*dy_hat - sum(dy_hat) - xhat*sum(dy_hat*xhat) )
// where xhat=(x-mean)*rstd

__global__ void layernorm_bwd_dx_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    const float* __restrict__ gamma, // may be nullptr => gamma=1
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ dx,
    int M, int N) {

  const int row = (int)blockIdx.x;
  if (row >= M) return;

  const float mu = mean[row];
  const float rs = rstd[row];

  const float* xr  = x  + (int64_t)row * (int64_t)N;
  const float* dyr = dy + (int64_t)row * (int64_t)N;
  float* dxr       = dx + (int64_t)row * (int64_t)N;

  float s1 = 0.0f; // sum(dy_hat)
  float s2 = 0.0f; // sum(dy_hat * xhat)
  for (int c = threadIdx.x; c < N; c += blockDim.x) {
    const float xv = xr[c];
    const float xhat = (xv - mu) * rs;
    float dyv = dyr[c];
    if (gamma) dyv *= gamma[c];
    s1 += dyv;
    s2 += dyv * xhat;
  }
  s1 = block_sum_f32(s1);
  s2 = block_sum_f32(s2);

  const float invN = 1.0f / (float)N;
  for (int c = threadIdx.x; c < N; c += blockDim.x) {
    const float xv = xr[c];
    const float xhat = (xv - mu) * rs;
    float dyv = dyr[c];
    if (gamma) dyv *= gamma[c];
    const float v = ( (float)N * dyv - s1 - xhat * s2 ) * (rs * invN);
    dxr[c] = v;
  }
}

__global__ void layernorm_bwd_dx_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ dy,
    const __half* __restrict__ gamma, // may be nullptr
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    __half* __restrict__ dx,
    int M, int N) {

  const int row = (int)blockIdx.x;
  if (row >= M) return;

  const float mu = mean[row];
  const float rs = rstd[row];

  const __half* xr  = x  + (int64_t)row * (int64_t)N;
  const __half* dyr = dy + (int64_t)row * (int64_t)N;
  __half* dxr       = dx + (int64_t)row * (int64_t)N;

  float s1 = 0.0f;
  float s2 = 0.0f;
  for (int c = threadIdx.x; c < N; c += blockDim.x) {
    const float xv = __half2float(xr[c]);
    const float xhat = (xv - mu) * rs;
    float dyv = __half2float(dyr[c]);
    if (gamma) dyv *= __half2float(gamma[c]);
    s1 += dyv;
    s2 += dyv * xhat;
  }
  s1 = block_sum_f32(s1);
  s2 = block_sum_f32(s2);

  const float invN = 1.0f / (float)N;
  for (int c = threadIdx.x; c < N; c += blockDim.x) {
    const float xv = __half2float(xr[c]);
    const float xhat = (xv - mu) * rs;
    float dyv = __half2float(dyr[c]);
    if (gamma) dyv *= __half2float(gamma[c]);
    const float v = ( (float)N * dyv - s1 - xhat * s2 ) * (rs * invN);
    dxr[c] = __float2half_rn(v);
  }
}

// -------------------- backward: dgamma/dbeta (f32 outputs) --------------------
// dbeta[c]  = sum_m dy[m,c]
// dgamma[c] = sum_m dy[m,c] * xhat[m,c]
// (xhat from x,mean,rstd)
// NOTE: This is simple and correct. Perf can be improved later.

__global__ void layernorm_bwd_dg_db_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int M, int N) {

  const int c = (int)blockIdx.x;
  if (c >= N) return;

  float sg = 0.0f;
  float sb = 0.0f;
  for (int r = threadIdx.x; r < M; r += blockDim.x) {
    const float mu = mean[r];
    const float rs = rstd[r];
    const float xv = x[(int64_t)r * (int64_t)N + c];
    const float xhat = (xv - mu) * rs;
    const float dyv = dy[(int64_t)r * (int64_t)N + c];
    sb += dyv;
    sg += dyv * xhat;
  }
  sg = block_sum_f32(sg);
  sb = block_sum_f32(sb);

  if (threadIdx.x == 0) {
    dgamma[c] = sg;
    dbeta[c]  = sb;
  }
}

__global__ void layernorm_bwd_dg_db_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ dy,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int M, int N) {

  const int c = (int)blockIdx.x;
  if (c >= N) return;

  float sg = 0.0f;
  float sb = 0.0f;
  for (int r = threadIdx.x; r < M; r += blockDim.x) {
    const float mu = mean[r];
    const float rs = rstd[r];
    const float xv = __half2float(x[(int64_t)r * (int64_t)N + c]);
    const float xhat = (xv - mu) * rs;
    const float dyv = __half2float(dy[(int64_t)r * (int64_t)N + c]);
    sb += dyv;
    sg += dyv * xhat;
  }
  sg = block_sum_f32(sg);
  sb = block_sum_f32(sb);

  if (threadIdx.x == 0) {
    dgamma[c] = sg;
    dbeta[c]  = sb;
  }
}

} // namespace layernorm_impl

// ============================================================================
// Variant: LayerNormFwd (f16/f32) -- you already use this in test_layernorm_fwd.py
// ============================================================================

static bool ln_fwd_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void*) {

  if (!inputs || !outputs) return false;
  if (!(num_inputs == 1 || num_inputs == 3)) return false;
  if (num_outputs != 3) return false;

  const auto& X = inputs[0];
  const auto& Y = outputs[0];
  const auto& Mean = outputs[1];
  const auto& Rstd = outputs[2];

  if (!is_f32_2d(X) || !is_f32_2d(Y)) return false;
  if (!is_contig_rowmajor_2d(X) || !is_contig_rowmajor_2d(Y)) return false;
  if (!(Y.shape[0] == X.shape[0] && Y.shape[1] == X.shape[1])) return false;

  if (!(Mean.rank()==1 && Mean.dtype==DType::kF32 && Mean.shape[0]==X.shape[0])) return false;
  if (!(Rstd.rank()==1 && Rstd.dtype==DType::kF32 && Rstd.shape[0]==X.shape[0])) return false;

  if (num_inputs == 3) {
    const auto& G = inputs[1];
    const auto& B = inputs[2];
    if (!(G.rank()==1 && B.rank()==1)) return false;
    if (!(G.dtype==DType::kF32 && B.dtype==DType::kF32)) return false;
    if (!(G.shape[0]==X.shape[1] && B.shape[0]==X.shape[1])) return false;
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

  const auto& X = inputs[0];
  auto& Y = outputs[0];
  auto& Mean = outputs[1];
  auto& Rstd = outputs[2];

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
    threads = 1; while (threads < N) threads <<= 1;
    if (threads > 256) threads = 256;
  }

  std::fprintf(stderr,
    "[LN f32] num_inputs=%d num_outputs=%d X=%p G=%p B=%p | Y=%p Mean=%p Rstd=%p | M=%d N=%d | eps=%g | threads=%d\n",
    num_inputs, num_outputs, (const void*)x, (const void*)gamma, (const void*)beta,
    (const void*)y, (const void*)mean, (const void*)rstd, M, N, (double)eps, threads);

  dim3 block(threads, 1, 1);
  dim3 grid(M, 1, 1);

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

static bool ln_fwd_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void*) {

  if (!inputs || !outputs) return false;
  if (!(num_inputs == 1 || num_inputs == 3)) return false;
  if (num_outputs != 3) return false;

  const auto& X = inputs[0];
  const auto& Y = outputs[0];
  const auto& Mean = outputs[1];
  const auto& Rstd = outputs[2];

  if (!is_f16_2d(X) || !is_f16_2d(Y)) return false;
  if (!is_contig_rowmajor_2d(X) || !is_contig_rowmajor_2d(Y)) return false;
  if (!(Y.shape[0] == X.shape[0] && Y.shape[1] == X.shape[1])) return false;

  if (!(Mean.rank()==1 && Mean.dtype==DType::kF32 && Mean.shape[0]==X.shape[0])) return false;
  if (!(Rstd.rank()==1 && Rstd.dtype==DType::kF32 && Rstd.shape[0]==X.shape[0])) return false;

  if (num_inputs == 3) {
    const auto& G = inputs[1];
    const auto& B = inputs[2];
    if (!(G.rank()==1 && B.rank()==1)) return false;
    if (!(G.dtype==DType::kF16 && B.dtype==DType::kF16)) return false;
    if (!(G.shape[0]==X.shape[1] && B.shape[0]==X.shape[1])) return false;
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

  const auto& X = inputs[0];
  auto& Y = outputs[0];
  auto& Mean = outputs[1];
  auto& Rstd = outputs[2];

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
    threads = 1; while (threads < N) threads <<= 1;
    if (threads > 256) threads = 256;
  }

  std::fprintf(stderr,
    "[LN f16] num_inputs=%d num_outputs=%d X=%p G=%p B=%p | Y=%p Mean=%p Rstd=%p | M=%d N=%d | eps=%g | threads=%d\n",
    num_inputs, num_outputs, (const void*)x, (const void*)gamma, (const void*)beta,
    (const void*)y, (const void*)mean, (const void*)rstd, M, N, (double)eps, threads);

  dim3 block(threads, 1, 1);
  dim3 grid(M, 1, 1);

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
// Variant: LayerNormBwd (f16/f32)
// attr: none for now
// ============================================================================

static bool ln_bwd_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void*) {

  if (!inputs || !outputs) return false;

  // affine: 5 inputs -> 3 outputs
  // noaff : 4 inputs -> 1 output
  const bool affine = (num_inputs == 5 && num_outputs == 3);
  const bool noaff  = (num_inputs == 4 && num_outputs == 1);
  if (!(affine || noaff)) return false;

  const auto& X  = inputs[0];
  const auto& dY = inputs[1];
  const int M = (int)X.shape[0];
  const int N = (int)X.shape[1];

  if (!is_f32_2d(X) || !is_f32_2d(dY)) return false;
  if (!is_contig_rowmajor_2d(X) || !is_contig_rowmajor_2d(dY)) return false;
  if (!(dY.shape[0]==X.shape[0] && dY.shape[1]==X.shape[1])) return false;

  const auto& Mean = inputs[affine ? 3 : 2];
  const auto& Rstd = inputs[affine ? 4 : 3];

  if (!(Mean.rank()==1 && Mean.dtype==DType::kF32 && Mean.shape[0]==M)) return false;
  if (!(Rstd.rank()==1 && Rstd.dtype==DType::kF32 && Rstd.shape[0]==M)) return false;

  const auto& dX = outputs[0];
  if (!is_f32_2d(dX) || !is_contig_rowmajor_2d(dX)) return false;
  if (!(dX.shape[0]==M && dX.shape[1]==N)) return false;

  if (affine) {
    const auto& G = inputs[2];
    if (!(G.rank()==1 && G.dtype==DType::kF32 && G.shape[0]==N)) return false;

    const auto& dG = outputs[1];
    const auto& dB = outputs[2];
    if (!(dG.rank()==1 && dG.dtype==DType::kF32 && dG.shape[0]==N)) return false;
    if (!(dB.rank()==1 && dB.dtype==DType::kF32 && dB.shape[0]==N)) return false;
  }

  return true;
}

static size_t ln_bwd_f32_workspace(const TensorDesc*, int, const void*) { return 0; }

static aicf::Status ln_bwd_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void*,
    void*, size_t,
    cudaStream_t stream) {

  if (!ln_bwd_f32_supported(inputs, num_inputs, outputs, num_outputs, nullptr)) {
    return aicf::Status::InvalidArgument;
  }

  const bool affine = (num_inputs == 5);

  const auto& X  = inputs[0];
  const auto& dY = inputs[1];

  const float* x  = (const float*)X.data;
  const float* dy = (const float*)dY.data;

  const float* gamma = affine ? (const float*)inputs[2].data : nullptr;

  const float* mean = (const float*)inputs[affine ? 3 : 2].data;
  const float* rstd = (const float*)inputs[affine ? 4 : 3].data;

  float* dx = (float*)outputs[0].data;

  const int M = (int)X.shape[0];
  const int N = (int)X.shape[1];

  int threads_row = 256;
  if (N < threads_row) {
    threads_row = 1; while (threads_row < N) threads_row <<= 1;
    if (threads_row > 256) threads_row = 256;
  }

  std::fprintf(stderr,
    "[LN bwd f32] affine=%d X=%p dY=%p G=%p mean=%p rstd=%p | dX=%p | M=%d N=%d | threads_row=%d\n",
    affine ? 1 : 0, (const void*)x, (const void*)dy, (const void*)gamma,
    (const void*)mean, (const void*)rstd, (void*)dx, M, N, threads_row);

  // dx
  {
    dim3 block(threads_row, 1, 1);
    dim3 grid(M, 1, 1);
    layernorm_impl::layernorm_bwd_dx_f32_kernel<<<grid, block, 0, stream>>>(
        x, dy, gamma, mean, rstd, dx, M, N);
    auto st = aicf::cuda::shim::cuda_last_error_to_status();
    if (!aicf::ok(st)) return st;
  }

  // dgamma/dbeta (only if affine)
  if (affine) {
    float* dgamma = (float*)outputs[1].data;
    float* dbeta  = (float*)outputs[2].data;

    // simple: one block per column
    // threads over M
    int threads = 256;
    if (M < threads) {
      threads = 1; while (threads < M) threads <<= 1;
      if (threads > 256) threads = 256;
    }

    std::fprintf(stderr,
      "[LN bwd f32 dg/db] dgamma=%p dbeta=%p | threads=%d grid(N)=%d\n",
      (void*)dgamma, (void*)dbeta, threads, N);

    dim3 block(threads, 1, 1);
    dim3 grid(N, 1, 1);
    layernorm_impl::layernorm_bwd_dg_db_f32_kernel<<<grid, block, 0, stream>>>(
        x, dy, mean, rstd, dgamma, dbeta, M, N);
    return aicf::cuda::shim::cuda_last_error_to_status();
  }

  return aicf::Status::Ok;
}

KernelVariant make_layernorm_bwd_f32_variant() {
  KernelVariant v{};
  v.name = "layernorm_bwd_f32_contig2d";
  v.priority = 0;
  v.flags = 0;
  v.launch = ln_bwd_f32_launch;
  v.supported = ln_bwd_f32_supported;
  v.query_workspace = ln_bwd_f32_workspace;
  return v;
}

static bool ln_bwd_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void*) {

  if (!inputs || !outputs) return false;

  const bool affine = (num_inputs == 5 && num_outputs == 3);
  const bool noaff  = (num_inputs == 4 && num_outputs == 1);
  if (!(affine || noaff)) return false;

  const auto& X  = inputs[0];
  const auto& dY = inputs[1];
  const int M = (int)X.shape[0];
  const int N = (int)X.shape[1];

  if (!is_f16_2d(X) || !is_f16_2d(dY)) return false;
  if (!is_contig_rowmajor_2d(X) || !is_contig_rowmajor_2d(dY)) return false;
  if (!(dY.shape[0]==X.shape[0] && dY.shape[1]==X.shape[1])) return false;

  const auto& Mean = inputs[affine ? 3 : 2];
  const auto& Rstd = inputs[affine ? 4 : 3];
  if (!(Mean.rank()==1 && Mean.dtype==DType::kF32 && Mean.shape[0]==M)) return false;
  if (!(Rstd.rank()==1 && Rstd.dtype==DType::kF32 && Rstd.shape[0]==M)) return false;

  const auto& dX = outputs[0];
  if (!is_f16_2d(dX) || !is_contig_rowmajor_2d(dX)) return false;
  if (!(dX.shape[0]==M && dX.shape[1]==N)) return false;

  if (affine) {
    const auto& G = inputs[2];
    if (!(G.rank()==1 && G.dtype==DType::kF16 && G.shape[0]==N)) return false;

    const auto& dG = outputs[1];
    const auto& dB = outputs[2];
    if (!(dG.rank()==1 && dG.dtype==DType::kF32 && dG.shape[0]==N)) return false;
    if (!(dB.rank()==1 && dB.dtype==DType::kF32 && dB.shape[0]==N)) return false;
  }
  return true;
}

static size_t ln_bwd_f16_workspace(const TensorDesc*, int, const void*) { return 0; }

static aicf::Status ln_bwd_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void*,
    void*, size_t,
    cudaStream_t stream) {

  if (!ln_bwd_f16_supported(inputs, num_inputs, outputs, num_outputs, nullptr)) {
    return aicf::Status::InvalidArgument;
  }

  const bool affine = (num_inputs == 5);

  const auto& X  = inputs[0];
  const auto& dY = inputs[1];

  const __half* x  = (const __half*)X.data;
  const __half* dy = (const __half*)dY.data;

  const __half* gamma = affine ? (const __half*)inputs[2].data : nullptr;

  const float* mean = (const float*)inputs[affine ? 3 : 2].data;
  const float* rstd = (const float*)inputs[affine ? 4 : 3].data;

  __half* dx = (__half*)outputs[0].data;

  const int M = (int)X.shape[0];
  const int N = (int)X.shape[1];

  int threads_row = 256;
  if (N < threads_row) {
    threads_row = 1; while (threads_row < N) threads_row <<= 1;
    if (threads_row > 256) threads_row = 256;
  }

  std::fprintf(stderr,
    "[LN bwd f16] affine=%d X=%p dY=%p G=%p mean=%p rstd=%p | dX=%p | M=%d N=%d | threads_row=%d\n",
    affine ? 1 : 0, (const void*)x, (const void*)dy, (const void*)gamma,
    (const void*)mean, (const void*)rstd, (void*)dx, M, N, threads_row);

  // dx
  {
    dim3 block(threads_row, 1, 1);
    dim3 grid(M, 1, 1);
    layernorm_impl::layernorm_bwd_dx_f16_kernel<<<grid, block, 0, stream>>>(
        x, dy, gamma, mean, rstd, dx, M, N);
    auto st = aicf::cuda::shim::cuda_last_error_to_status();
    if (!aicf::ok(st)) return st;
  }

  // dgamma/dbeta only if affine
  if (affine) {
    float* dgamma = (float*)outputs[1].data;
    float* dbeta  = (float*)outputs[2].data;

    int threads = 256;
    if (M < threads) {
      threads = 1; while (threads < M) threads <<= 1;
      if (threads > 256) threads = 256;
    }

    std::fprintf(stderr,
      "[LN bwd f16 dg/db] dgamma=%p dbeta=%p | threads=%d grid(N)=%d\n",
      (void*)dgamma, (void*)dbeta, threads, N);

    dim3 block(threads, 1, 1);
    dim3 grid(N, 1, 1);
    layernorm_impl::layernorm_bwd_dg_db_f16_kernel<<<grid, block, 0, stream>>>(
        x, dy, mean, rstd, dgamma, dbeta, M, N);
    return aicf::cuda::shim::cuda_last_error_to_status();
  }

  return aicf::Status::Ok;
}

KernelVariant make_layernorm_bwd_f16_variant() {
  KernelVariant v{};
  v.name = "layernorm_bwd_f16_contig2d";
  v.priority = 10;
  v.flags = 0;
  v.launch = ln_bwd_f16_launch;
  v.supported = ln_bwd_f16_supported;
  v.query_workspace = ln_bwd_f16_workspace;
  return v;
}

// ============================================================================
// C++ API entrypoints (raw pointer path) -- optional
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
  if (N < threads) { threads = 1; while (threads < N) threads <<= 1; if (threads > 256) threads = 256; }

  dim3 block(threads, 1, 1);
  dim3 grid(M, 1, 1);
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
  if (N < threads) { threads = 1; while (threads < N) threads <<= 1; if (threads > 256) threads = 256; }

  dim3 block(threads, 1, 1);
  dim3 grid(M, 1, 1);
  layernorm_impl::layernorm_fwd_f16_kernel<<<grid, block, 0, st>>>(
      (const __half*)x, (const __half*)gamma, (const __half*)beta,
      (__half*)y, mean, rstd, M, N, eps);
  return aicf::cuda::shim::cuda_last_error_to_status();
}

Status layernorm_bwd_f32(float* dx, float* dgamma, float* dbeta,
                         const float* x, const float* dy,
                         const float* gamma,
                         const float* mean, const float* rstd,
                         int M, int N,
                         Stream stream) {
  if (!dx || !x || !dy || !mean || !rstd) return Status::InvalidArgument;
  cudaStream_t st = unwrap(stream);

  int threads_row = 256;
  if (N < threads_row) { threads_row = 1; while (threads_row < N) threads_row <<= 1; if (threads_row > 256) threads_row = 256; }
  dim3 block1(threads_row, 1, 1);
  dim3 grid1(M, 1, 1);
  layernorm_impl::layernorm_bwd_dx_f32_kernel<<<grid1, block1, 0, st>>>(
      x, dy, gamma, mean, rstd, dx, M, N);
  auto s0 = aicf::cuda::shim::cuda_last_error_to_status();
  if (!aicf::ok(s0)) return s0;

  if (gamma && dgamma && dbeta) {
    int threads = 256;
    if (M < threads) { threads = 1; while (threads < M) threads <<= 1; if (threads > 256) threads = 256; }
    dim3 block2(threads, 1, 1);
    dim3 grid2(N, 1, 1);
    layernorm_impl::layernorm_bwd_dg_db_f32_kernel<<<grid2, block2, 0, st>>>(
        x, dy, mean, rstd, dgamma, dbeta, M, N);
    return aicf::cuda::shim::cuda_last_error_to_status();
  }
  return Status::Ok;
}

Status layernorm_bwd_f16(void* dx, float* dgamma, float* dbeta,
                         const void* x, const void* dy,
                         const void* gamma,
                         const float* mean, const float* rstd,
                         int M, int N,
                         Stream stream) {
  if (!dx || !x || !dy || !mean || !rstd) return Status::InvalidArgument;
  cudaStream_t st = unwrap(stream);

  int threads_row = 256;
  if (N < threads_row) { threads_row = 1; while (threads_row < N) threads_row <<= 1; if (threads_row > 256) threads_row = 256; }
  dim3 block1(threads_row, 1, 1);
  dim3 grid1(M, 1, 1);
  layernorm_impl::layernorm_bwd_dx_f16_kernel<<<grid1, block1, 0, st>>>(
      (const __half*)x, (const __half*)dy, (const __half*)gamma, mean, rstd, (__half*)dx, M, N);
  auto s0 = aicf::cuda::shim::cuda_last_error_to_status();
  if (!aicf::ok(s0)) return s0;

  if (gamma && dgamma && dbeta) {
    int threads = 256;
    if (M < threads) { threads = 1; while (threads < M) threads <<= 1; if (threads > 256) threads = 256; }
    dim3 block2(threads, 1, 1);
    dim3 grid2(N, 1, 1);
    layernorm_impl::layernorm_bwd_dg_db_f16_kernel<<<grid2, block2, 0, st>>>(
        (const __half*)x, (const __half*)dy, mean, rstd, dgamma, dbeta, M, N);
    return aicf::cuda::shim::cuda_last_error_to_status();
  }
  return Status::Ok;
}

} // namespace aicf::cuda
