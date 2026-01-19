// ============================================================================
// src/backends/cuda/ops/layernorm/launcher.cu  (core-free / minimal)
// - kernel definitions live in this file
// - 2D only: x[M,N] contiguous row-major, normalize over last dim (N)
// - Fwd: inputs = (X) or (X,G,B) ; outputs = (Y, Mean[f32,M], Rstd[f32,M])
// - Bwd: noaff  inputs=(X, dY, Mean, Rstd)             outputs=(dX)
//        affine inputs=(X, dY, G, Mean, Rstd)           outputs=(dX, dG[f32,N], dB[f32,N])
// - AttrBlob: eps optional (schema 'LNEP' or schema_id==0 => default eps)
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <cmath>

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
// AttrBlob schema: eps (float32)
// schema_id: 'LNEP' = 0x50454E4C (little-endian: 'L''N''E''P')
// schema_id==0 allowed -> default eps
// -------------------------
static constexpr uint32_t kSchema_LNEP = 0x50454E4Cu;

static inline float read_f32_le(const uint8_t* p) {
  float v;
  std::memcpy(&v, p, sizeof(float));
  return v;
}

static inline float get_eps_from_attr(const void* attr, float default_eps) {
  if (!attr) return default_eps;
  const AttrBlob& a = *static_cast<const AttrBlob*>(attr);
  if (a.schema_id == 0) return default_eps;
  if (a.schema_id != kSchema_LNEP) return default_eps;
  if (!a.data || a.bytes < 4) return default_eps;
  return read_f32_le(static_cast<const uint8_t*>(a.data));
}

// -------------------------
// Tensor helpers (2D contiguous row-major only)
// -------------------------
static inline bool is_2d(const TensorDesc& T) { return T.rank() == 2; }

static inline bool is_contig_rowmajor_2d(const TensorDesc& T) {
  if (!is_2d(T)) return false;
  if (!T.contiguous) return false;
  // assume TensorDesc stride is valid if contiguous; if you do store stride:
  // require stride[1]==1 and stride[0]==shape[1]
  if (T.stride[1] != 1) return false;
  if (T.stride[0] != T.shape[1]) return false;
  return true;
}

static inline bool is_f32_2d_contig(const TensorDesc& T) {
  return (T.dtype == DType::kF32) && is_contig_rowmajor_2d(T);
}
static inline bool is_f16_2d_contig(const TensorDesc& T) {
  return (T.dtype == DType::kF16) && is_contig_rowmajor_2d(T);
}

static inline bool is_f32_1d_len(const TensorDesc& T, int64_t n) {
  return (T.dtype == DType::kF32) && (T.rank() == 1) && T.contiguous && (T.shape[0] == n);
}
static inline bool is_f16_1d_len(const TensorDesc& T, int64_t n) {
  return (T.dtype == DType::kF16) && (T.rank() == 1) && T.contiguous && (T.shape[0] == n);
}

static inline bool is_f32_1d_M(const TensorDesc& T, int64_t M) {
  return is_f32_1d_len(T, M);
}

static inline bool same_shape_2d(const TensorDesc& A, const TensorDesc& B) {
  return (A.rank() == 2 && B.rank() == 2 && A.shape[0] == B.shape[0] && A.shape[1] == B.shape[1]);
}

// -------------------------
// kernels (definitions live here)
// -------------------------
namespace layernorm_impl {

__device__ __forceinline__ float warp_sum_f32(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

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

// ---- forward ----

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

// ---- backward dx ----
// dy_hat = dy * gamma (gamma==nullptr => 1)
// dx = (1/N)*rstd * ( N*dy_hat - sum(dy_hat) - xhat*sum(dy_hat*xhat) )
__global__ void layernorm_bwd_dx_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    const float* __restrict__ gamma,
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

  float s1 = 0.0f;
  float s2 = 0.0f;
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
    dxr[c] = (((float)N * dyv - s1 - xhat * s2) * (rs * invN));
  }
}

__global__ void layernorm_bwd_dx_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ dy,
    const __half* __restrict__ gamma,
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
    const float v = (((float)N * dyv - s1 - xhat * s2) * (rs * invN));
    dxr[c] = __float2half_rn(v);
  }
}

// ---- backward dgamma/dbeta (float outputs) ----
// dbeta[c]  = sum_m dy[m,c]
// dgamma[c] = sum_m dy[m,c] * xhat[m,c]
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

// -------------------------
// launch config helper
// -------------------------
static inline int pick_threads_pow2_le_256(int n) {
  int t = 256;
  if (n < t) {
    t = 1;
    while (t < n) t <<= 1;
    if (t > 256) t = 256;
  }
  return t;
}

static inline int pick_blocks_1d(int64_t n, int threads) {
  int64_t b = (n + threads - 1) / threads;
  if (b < 1) b = 1;
  if (b > 65535) b = 65535;
  return (int)b;
}

// ============================================================================
// LayerNormFwd variants
// ============================================================================

static inline bool ln_fwd_check_common(
    const TensorDesc* inputs, int ni,
    const TensorDesc* outputs, int no,
    bool (*is_x_ok)(const TensorDesc&),
    bool (*is_gb_ok)(const TensorDesc&, int64_t)) {

  if (!inputs || !outputs) return false;
  if (!(ni == 1 || ni == 3)) return false;
  if (no != 3) return false;

  const TensorDesc& X = inputs[0];
  const TensorDesc& Y = outputs[0];
  const TensorDesc& Mean = outputs[1];
  const TensorDesc& Rstd = outputs[2];

  if (!is_x_ok(X) || !is_x_ok(Y)) return false;
  if (!same_shape_2d(X, Y)) return false;

  const int64_t M = X.shape[0];
  const int64_t N = X.shape[1];

  if (!is_f32_1d_M(Mean, M)) return false;
  if (!is_f32_1d_M(Rstd, M)) return false;

  if (ni == 3) {
    const TensorDesc& G = inputs[1];
    const TensorDesc& B = inputs[2];
    if (!is_gb_ok(G, N) || !is_gb_ok(B, N)) return false;
  }
  return true;
}

static bool ln_fwd_f32_supported(const TensorDesc* in, int ni, const TensorDesc* out, int no, const void*) {
  return ln_fwd_check_common(in, ni, out, no, &is_f32_2d_contig, &is_f32_1d_len);
}

static Status ln_fwd_f32_launch(
    const TensorDesc* inputs, int ni,
    TensorDesc* outputs, int no,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!ln_fwd_f32_supported(inputs, ni, outputs, no, attr)) return Status::InvalidArgument;

  const TensorDesc& X = inputs[0];
  TensorDesc& Y = outputs[0];
  TensorDesc& Mean = outputs[1];
  TensorDesc& Rstd = outputs[2];

  const float* gamma = nullptr;
  const float* beta = nullptr;
  if (ni == 3) { gamma = (const float*)inputs[1].data; beta = (const float*)inputs[2].data; }

  const int M = (int)X.shape[0];
  const int N = (int)X.shape[1];

  const float eps = get_eps_from_attr(attr, 1e-5f);
  const int threads = pick_threads_pow2_le_256(N);

  cudaGetLastError(); // clear
  layernorm_impl::layernorm_fwd_f32_kernel<<<dim3(M,1,1), dim3(threads,1,1), 0, stream>>>(
      (const float*)X.data, gamma, beta,
      (float*)Y.data,
      (float*)Mean.data, (float*)Rstd.data,
      M, N, eps);

  return cuda_last_status();
}

static size_t ln_fwd_ws(const TensorDesc*, int, const void*) { return 0; }

KernelVariant make_layernorm_fwd_f32_variant() {
  KernelVariant v{};
  v.name = "layernorm_fwd_f32_contig2d";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = kSchema_LNEP; // schema_id==0 also ok if dispatcher allows
  v.launch = ln_fwd_f32_launch;
  v.supported = ln_fwd_f32_supported;
  v.query_workspace = ln_fwd_ws;
  return v;
}

static bool ln_fwd_f16_supported(const TensorDesc* in, int ni, const TensorDesc* out, int no, const void*) {
  auto gb_ok = [](const TensorDesc& T, int64_t n) { return is_f16_1d_len(T, n); };
  return ln_fwd_check_common(in, ni, out, no, &is_f16_2d_contig, gb_ok);
}

static Status ln_fwd_f16_launch(
    const TensorDesc* inputs, int ni,
    TensorDesc* outputs, int no,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!ln_fwd_f16_supported(inputs, ni, outputs, no, attr)) return Status::InvalidArgument;

  const TensorDesc& X = inputs[0];
  TensorDesc& Y = outputs[0];
  TensorDesc& Mean = outputs[1];
  TensorDesc& Rstd = outputs[2];

  const __half* gamma = nullptr;
  const __half* beta = nullptr;
  if (ni == 3) { gamma = (const __half*)inputs[1].data; beta = (const __half*)inputs[2].data; }

  const int M = (int)X.shape[0];
  const int N = (int)X.shape[1];

  const float eps = get_eps_from_attr(attr, 1e-5f);
  const int threads = pick_threads_pow2_le_256(N);

  cudaGetLastError(); // clear
  layernorm_impl::layernorm_fwd_f16_kernel<<<dim3(M,1,1), dim3(threads,1,1), 0, stream>>>(
      (const __half*)X.data, gamma, beta,
      (__half*)Y.data,
      (float*)Mean.data, (float*)Rstd.data,
      M, N, eps);

  return cuda_last_status();
}

KernelVariant make_layernorm_fwd_f16_variant() {
  KernelVariant v{};
  v.name = "layernorm_fwd_f16_contig2d";
  v.priority = 10;
  v.flags = 0;
  v.expected_attr_schema_id = kSchema_LNEP;
  v.launch = ln_fwd_f16_launch;
  v.supported = ln_fwd_f16_supported;
  v.query_workspace = ln_fwd_ws;
  return v;
}

// ============================================================================
// LayerNormBwd variants
// ============================================================================

static bool ln_bwd_f32_supported(const TensorDesc* in, int ni, const TensorDesc* out, int no, const void*) {
  if (!in || !out) return false;

  const bool affine = (ni == 5 && no == 3);
  const bool noaff  = (ni == 4 && no == 1);
  if (!(affine || noaff)) return false;

  const TensorDesc& X  = in[0];
  const TensorDesc& dY = in[1];
  if (!is_f32_2d_contig(X) || !is_f32_2d_contig(dY)) return false;
  if (!same_shape_2d(X, dY)) return false;

  const int64_t M = X.shape[0];
  const int64_t N = X.shape[1];

  const TensorDesc& Mean = in[affine ? 3 : 2];
  const TensorDesc& Rstd = in[affine ? 4 : 3];
  if (!is_f32_1d_M(Mean, M)) return false;
  if (!is_f32_1d_M(Rstd, M)) return false;

  const TensorDesc& dX = out[0];
  if (!is_f32_2d_contig(dX)) return false;
  if (!same_shape_2d(X, dX)) return false;

  if (affine) {
    const TensorDesc& G = in[2];
    if (!is_f32_1d_len(G, N)) return false;

    const TensorDesc& dG = out[1];
    const TensorDesc& dB = out[2];
    if (!is_f32_1d_len(dG, N)) return false;
    if (!is_f32_1d_len(dB, N)) return false;
  }
  return true;
}

static Status ln_bwd_f32_launch(
    const TensorDesc* in, int ni,
    TensorDesc* out, int no,
    const void*,
    void*, size_t,
    cudaStream_t stream) {

  if (!ln_bwd_f32_supported(in, ni, out, no, nullptr)) return Status::InvalidArgument;

  const bool affine = (ni == 5);

  const TensorDesc& X  = in[0];
  const TensorDesc& dY = in[1];

  const float* gamma = affine ? (const float*)in[2].data : nullptr;
  const float* mean  = (const float*)in[affine ? 3 : 2].data;
  const float* rstd  = (const float*)in[affine ? 4 : 3].data;

  const int M = (int)X.shape[0];
  const int N = (int)X.shape[1];

  const int threads_row = pick_threads_pow2_le_256(N);

  cudaGetLastError(); // clear
  layernorm_impl::layernorm_bwd_dx_f32_kernel<<<dim3(M,1,1), dim3(threads_row,1,1), 0, stream>>>(
      (const float*)X.data,
      (const float*)dY.data,
      gamma,
      mean, rstd,
      (float*)out[0].data,
      M, N);

  auto st0 = cuda_last_status();
  if (st0 != Status::Ok) return st0;

  if (affine) {
    // one block per column, threads over M
    int threads = pick_threads_pow2_le_256(M);
    cudaGetLastError(); // clear
    layernorm_impl::layernorm_bwd_dg_db_f32_kernel<<<dim3(N,1,1), dim3(threads,1,1), 0, stream>>>(
        (const float*)X.data,
        (const float*)dY.data,
        mean, rstd,
        (float*)out[1].data,
        (float*)out[2].data,
        M, N);
    return cuda_last_status();
  }
  return Status::Ok;
}

static bool ln_bwd_f16_supported(const TensorDesc* in, int ni, const TensorDesc* out, int no, const void*) {
  if (!in || !out) return false;

  const bool affine = (ni == 5 && no == 3);
  const bool noaff  = (ni == 4 && no == 1);
  if (!(affine || noaff)) return false;

  const TensorDesc& X  = in[0];
  const TensorDesc& dY = in[1];
  if (!is_f16_2d_contig(X) || !is_f16_2d_contig(dY)) return false;
  if (!same_shape_2d(X, dY)) return false;

  const int64_t M = X.shape[0];
  const int64_t N = X.shape[1];

  const TensorDesc& Mean = in[affine ? 3 : 2];
  const TensorDesc& Rstd = in[affine ? 4 : 3];
  if (!is_f32_1d_M(Mean, M)) return false;
  if (!is_f32_1d_M(Rstd, M)) return false;

  const TensorDesc& dX = out[0];
  if (!is_f16_2d_contig(dX)) return false;
  if (!same_shape_2d(X, dX)) return false;

  if (affine) {
    const TensorDesc& G = in[2];
    if (!is_f16_1d_len(G, N)) return false;

    const TensorDesc& dG = out[1];
    const TensorDesc& dB = out[2];
    if (!is_f32_1d_len(dG, N)) return false;
    if (!is_f32_1d_len(dB, N)) return false;
  }
  return true;
}

static Status ln_bwd_f16_launch(
    const TensorDesc* in, int ni,
    TensorDesc* out, int no,
    const void*,
    void*, size_t,
    cudaStream_t stream) {

  if (!ln_bwd_f16_supported(in, ni, out, no, nullptr)) return Status::InvalidArgument;

  const bool affine = (ni == 5);

  const TensorDesc& X  = in[0];
  const TensorDesc& dY = in[1];

  const __half* gamma = affine ? (const __half*)in[2].data : nullptr;
  const float* mean   = (const float*)in[affine ? 3 : 2].data;
  const float* rstd   = (const float*)in[affine ? 4 : 3].data;

  const int M = (int)X.shape[0];
  const int N = (int)X.shape[1];

  const int threads_row = pick_threads_pow2_le_256(N);

  cudaGetLastError(); // clear
  layernorm_impl::layernorm_bwd_dx_f16_kernel<<<dim3(M,1,1), dim3(threads_row,1,1), 0, stream>>>(
      (const __half*)X.data,
      (const __half*)dY.data,
      gamma,
      mean, rstd,
      (__half*)out[0].data,
      M, N);

  auto st0 = cuda_last_status();
  if (st0 != Status::Ok) return st0;

  if (affine) {
    int threads = pick_threads_pow2_le_256(M);
    cudaGetLastError(); // clear
    layernorm_impl::layernorm_bwd_dg_db_f16_kernel<<<dim3(N,1,1), dim3(threads,1,1), 0, stream>>>(
        (const __half*)X.data,
        (const __half*)dY.data,
        mean, rstd,
        (float*)out[1].data,
        (float*)out[2].data,
        M, N);
    return cuda_last_status();
  }
  return Status::Ok;
}

static size_t ln_bwd_ws(const TensorDesc*, int, const void*) { return 0; }

KernelVariant make_layernorm_bwd_f32_variant() {
  KernelVariant v{};
  v.name = "layernorm_bwd_f32_contig2d";
  v.priority = 0;
  v.flags = 0;
  v.launch = ln_bwd_f32_launch;
  v.supported = ln_bwd_f32_supported;
  v.query_workspace = ln_bwd_ws;
  return v;
}

KernelVariant make_layernorm_bwd_f16_variant() {
  KernelVariant v{};
  v.name = "layernorm_bwd_f16_contig2d";
  v.priority = 10;
  v.flags = 0;
  v.launch = ln_bwd_f16_launch;
  v.supported = ln_bwd_f16_supported;
  v.query_workspace = ln_bwd_ws;
  return v;
}

} // namespace aicf::cuda
