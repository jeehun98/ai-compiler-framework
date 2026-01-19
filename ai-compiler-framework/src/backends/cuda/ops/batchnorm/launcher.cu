// ============================================================================
// src/backends/cuda/ops/batchnorm/launcher.cu
// - KEEP "kernel definitions inside launcher.cu" structure
// - NCHW 4D only: x[N,C,H,W], normalize per-channel over N*H*W
// - dtype: f16 only (x/y/gamma/beta f16), stats f32
// - Forward:
//    inference: y only (use running_mean/var)
//    training : y + save_mean(C) + save_rstd(C) (batch stats)
//      save_mean used as sum->mean
//      save_rstd used as sumsq->var then in-place var->rstd
// - Backward (training):
//    affine=True : outputs dx + dgamma(f32) + dbeta(f32)
//    affine=False: NotImplemented
// - Correctness-first: atomic reductions
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>

#include <aicf/backends/cuda/registry/status.hpp>
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_blob.hpp>

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
// AttrBlob schema: BatchNorm
// schema_id: 'BNEP' (0x50454E42)  // BatchNorm Eps + flags
// payload (bytes):
//   f32 eps
//   u32 flags  (bit0=use_running_stats)
// NOTE:
// - update_running reserved, ignored
// - if schema_id==0 => default eps=1e-5, use_running_stats=true
// -------------------------
static constexpr uint32_t kSchema_BNEP = 0x50454E42u;

static inline float read_f32_le(const uint8_t* p) {
  float v;
  std::memcpy(&v, p, sizeof(float));
  return v;
}
static inline uint32_t read_u32_le(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, sizeof(uint32_t));
  return v;
}

struct BNAttr {
  float eps;
  bool use_running_stats;
};

static inline BNAttr get_bn_attr(const AttrBlob* ab) {
  BNAttr a{};
  a.eps = 1e-5f;
  a.use_running_stats = true;

  if (!ab) return a;
  if (ab->schema_id == 0) return a;
  if (ab->schema_id != kSchema_BNEP) return a;
  if (!ab->data || ab->bytes < 8) return a;

  const auto* p = static_cast<const uint8_t*>(ab->data);
  a.eps = read_f32_le(p + 0);
  const uint32_t flags = read_u32_le(p + 4);
  a.use_running_stats = (flags & 1u) != 0;
  return a;
}

// ---------------- tensor helpers ----------------
static inline bool is_4d(const TensorDesc& T) { return T.rank() == 4; }
static inline bool stride_valid_4d(const TensorDesc& T) {
  if (!is_4d(T)) return false;
  return (T.stride[0] > 0 && T.stride[1] > 0 && T.stride[2] > 0 && T.stride[3] > 0);
}
static inline bool is_contig_nchw_4d(const TensorDesc& T) {
  if (!is_4d(T) || !stride_valid_4d(T)) return false;
  const int64_t C = T.shape[1];
  const int64_t H = T.shape[2];
  const int64_t W = T.shape[3];
  return (T.stride[3] == 1) &&
         (T.stride[2] == W) &&
         (T.stride[1] == H * W) &&
         (T.stride[0] == C * H * W);
}
static inline bool is_f16_4d(const TensorDesc& T) { return (T.dtype == DType::kF16) && is_4d(T); }
static inline bool is_f16_1d(const TensorDesc& T) { return (T.dtype == DType::kF16) && (T.rank() == 1); }
static inline bool is_f32_1d(const TensorDesc& T) { return (T.dtype == DType::kF32) && (T.rank() == 1); }

// ============================================================================
// kernels (definitions) -- keep here
// ============================================================================

namespace bn_impl {

// sum/sumsq into float[C] (atomic)
__global__ void bn_fwd_stats_f16_atomic(
    const __half* __restrict__ x,
    float* __restrict__ sum,
    float* __restrict__ sumsq,
    int N, int C, int HW) {

  const int64_t total = (int64_t)N * (int64_t)C * (int64_t)HW;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < total;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int c = (int)((i / HW) % C);
    const float v = __half2float(x[i]);
    atomicAdd(&sum[c], v);
    atomicAdd(&sumsq[c], v * v);
  }
}

// finalize: sum->mean, sumsq->var
__global__ void bn_finalize_mean_var(
    float* __restrict__ mean,   // in: sum, out: mean
    float* __restrict__ var,    // in: sumsq, out: var
    int C,
    float invNHW) {

  const int c = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (c >= C) return;

  float m = mean[c] * invNHW;
  float v = var[c]  * invNHW - m * m;
  if (v < 0.0f) v = 0.0f;
  mean[c] = m;
  var[c]  = v;
}

// in-place: var -> rstd
__global__ void bn_var_to_rstd_inplace(
    float* __restrict__ var_inplace, // becomes rstd
    int C,
    float eps) {

  const int c = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (c >= C) return;
  var_inplace[c] = rsqrtf(var_inplace[c] + eps);
}

// apply: y = ((x - mean) * rstd) * gamma + beta
__global__ void bn_fwd_apply_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma, // nullable
    const __half* __restrict__ beta,  // nullable
    const float* __restrict__ mean,   // [C]
    const float* __restrict__ rstd,   // [C]
    __half* __restrict__ y,
    int N, int C, int HW) {

  const int64_t total = (int64_t)N * (int64_t)C * (int64_t)HW;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < total;
       i += (int64_t)gridDim.x * blockDim.x) {

    const int c = (int)((i / HW) % C);
    const float mu = mean[c];
    const float rs = rstd[c];

    float g = 1.0f;
    float b = 0.0f;
    if (gamma) g = __half2float(gamma[c]);
    if (beta)  b = __half2float(beta[c]);

    const float xv = __half2float(x[i]);
    const float xhat = (xv - mu) * rs;
    const float out = xhat * g + b;
    y[i] = __float2half_rn(out);
  }
}

// inference apply: mean/var given (var -> rsqrt(var+eps) inside)
__global__ void bn_infer_apply_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    __half* __restrict__ y,
    int N, int C, int HW,
    float eps) {

  const int64_t total = (int64_t)N * (int64_t)C * (int64_t)HW;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < total;
       i += (int64_t)gridDim.x * blockDim.x) {

    const int c = (int)((i / HW) % C);
    const float mu = mean[c];
    const float rs = rsqrtf(var[c] + eps);

    float g = 1.0f;
    float b = 0.0f;
    if (gamma) g = __half2float(gamma[c]);
    if (beta)  b = __half2float(beta[c]);

    const float xv = __half2float(x[i]);
    const float xhat = (xv - mu) * rs;
    const float out = xhat * g + b;
    y[i] = __float2half_rn(out);
  }
}

// ---- BWD sums (PyTorch) ----
// dbeta[c]  = sum(dy)
// dgamma[c] = sum(dy * xhat)
__global__ void bn_bwd_sums_f16_atomic(
    const __half* __restrict__ x,
    const __half* __restrict__ dy,
    const float* __restrict__ mean,   // [C]
    const float* __restrict__ rstd,   // [C]
    float* __restrict__ sum_dy,       // [C] -> dbeta
    float* __restrict__ sum_dy_xhat,  // [C] -> dgamma
    int N, int C, int HW) {

  const int64_t total = (int64_t)N * (int64_t)C * (int64_t)HW;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < total;
       i += (int64_t)gridDim.x * blockDim.x) {

    const int c = (int)((i / HW) % C);
    const float mu = mean[c];
    const float rs = rstd[c];

    const float xv = __half2float(x[i]);
    const float xhat = (xv - mu) * rs;

    const float dyv = __half2float(dy[i]);
    atomicAdd(&sum_dy[c], dyv);
    atomicAdd(&sum_dy_xhat[c], dyv * xhat);
  }
}

// dx = (gamma * rstd / NHW) * ( NHW*dy - sum(dy) - xhat*sum(dy*xhat) )
__global__ void bn_bwd_dx_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ dy,
    const __half* __restrict__ gamma, // non-null
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ sum_dy,       // [C]
    const float* __restrict__ sum_dy_xhat,  // [C]
    __half* __restrict__ dx,
    int N, int C, int HW) {

  const int64_t total = (int64_t)N * (int64_t)C * (int64_t)HW;
  const float invNHW = 1.0f / (float)(N * HW);
  const float NHWf   = (float)(N * HW);

  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < total;
       i += (int64_t)gridDim.x * blockDim.x) {

    const int c = (int)((i / HW) % C);
    const float mu = mean[c];
    const float rs = rstd[c];

    const float xv = __half2float(x[i]);
    const float xhat = (xv - mu) * rs;

    const float dyv = __half2float(dy[i]);

    const float s1 = sum_dy[c];
    const float s2 = sum_dy_xhat[c];
    const float g  = __half2float(gamma[c]);

    const float v = (NHWf * dyv - s1 - xhat * s2) * (g * rs * invNHW);
    dx[i] = __float2half_rn(v);
  }
}

} // namespace bn_impl

// ============================================================================
// Supported checks
// ============================================================================

static bool bn_fwd_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void*) {

  if (!inputs || !outputs) return false;

  // inference:
  //  affine: inputs 5 => x,g,b,rm,rv ; outputs 1 => y
  //  noaff : inputs 3 => x,rm,rv     ; outputs 1 => y
  // training:
  //  affine: inputs 3 => x,g,b       ; outputs 3 => y,save_mean,save_rstd
  //  noaff : inputs 1 => x           ; outputs 3 => y,save_mean,save_rstd
  const bool inf_aff  = (num_inputs == 5 && num_outputs == 1);
  const bool inf_no   = (num_inputs == 3 && num_outputs == 1);
  const bool tr_aff   = (num_inputs == 3 && num_outputs == 3);
  const bool tr_no    = (num_inputs == 1 && num_outputs == 3);
  if (!(inf_aff || inf_no || tr_aff || tr_no)) return false;

  const auto& X = inputs[0];
  const auto& Y = outputs[0];

  if (!is_f16_4d(X) || !is_f16_4d(Y)) return false;
  if (!is_contig_nchw_4d(X) || !is_contig_nchw_4d(Y)) return false;
  for (int i = 0; i < 4; ++i) if (Y.shape[i] != X.shape[i]) return false;

  const int64_t C = X.shape[1];

  if (inf_aff || tr_aff) {
    const auto& G = inputs[1];
    const auto& B = inputs[2];
    if (!is_f16_1d(G) || !is_f16_1d(B)) return false;
    if (G.shape[0] != C || B.shape[0] != C) return false;
  }

  if (inf_aff || inf_no) {
    const auto& RM = inputs[inf_aff ? 3 : 1];
    const auto& RV = inputs[inf_aff ? 4 : 2];
    if (!is_f32_1d(RM) || !is_f32_1d(RV)) return false;
    if (RM.shape[0] != C || RV.shape[0] != C) return false;
  } else {
    const auto& SM = outputs[1];
    const auto& SR = outputs[2];
    if (!is_f32_1d(SM) || !is_f32_1d(SR)) return false;
    if (SM.shape[0] != C || SR.shape[0] != C) return false;
  }

  return true;
}

static size_t bn_fwd_f16_workspace(const TensorDesc*, int, const void*) { return 0; }

static Status bn_fwd_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!bn_fwd_f16_supported(inputs, num_inputs, outputs, num_outputs, nullptr)) {
    return Status::InvalidArgument;
  }

  const AttrBlob* ab = attr ? static_cast<const AttrBlob*>(attr) : nullptr;
  const BNAttr A = get_bn_attr(ab);

  const bool training = (num_outputs == 3);

  // strict: if training signature used, attrs must not force running stats
  if (training && A.use_running_stats) {
    return Status::InvalidArgument;
  }

  const bool inf_aff = (num_inputs == 5 && num_outputs == 1);
  const bool inf_no  = (num_inputs == 3 && num_outputs == 1);

  const auto& X = inputs[0];
  auto& Y = outputs[0];

  const int N  = (int)X.shape[0];
  const int C  = (int)X.shape[1];
  const int H  = (int)X.shape[2];
  const int W  = (int)X.shape[3];
  const int HW = H * W;

  const __half* x = (const __half*)X.data;
  __half* y = (__half*)Y.data;

  const __half* gamma = nullptr;
  const __half* beta  = nullptr;

  const float* running_mean = nullptr;
  const float* running_var  = nullptr;

  float* save_mean = nullptr; // training: sum -> mean
  float* save_rstd = nullptr; // training: sumsq -> var -> rstd

  if (inf_aff) {
    gamma = (const __half*)inputs[1].data;
    beta  = (const __half*)inputs[2].data;
    running_mean = (const float*)inputs[3].data;
    running_var  = (const float*)inputs[4].data;
  } else if (inf_no) {
    running_mean = (const float*)inputs[1].data;
    running_var  = (const float*)inputs[2].data;
  } else if (training) {
    if (num_inputs == 3) {
      gamma = (const __half*)inputs[1].data;
      beta  = (const __half*)inputs[2].data;
    }
    save_mean = (float*)outputs[1].data;
    save_rstd = (float*)outputs[2].data;
  }

  std::fprintf(stderr,
    "[BN fwd f16] in=%d out=%d training=%d X=%p G=%p B=%p RM=%p RV=%p | Y=%p SM=%p SR=%p | N=%d C=%d H=%d W=%d eps=%g use_running=%d\n",
    num_inputs, num_outputs, training ? 1 : 0,
    (const void*)x, (const void*)gamma, (const void*)beta,
    (const void*)running_mean, (const void*)running_var,
    (void*)y, (void*)save_mean, (void*)save_rstd,
    N, C, H, W, (double)A.eps, A.use_running_stats ? 1 : 0);

  const int threads = 256;

  // ---------------- training path ----------------
  if (training) {
    float* sum   = save_mean; // [C]
    float* sumsq = save_rstd; // [C] -> var -> rstd

    cudaError_t e0 = cudaMemsetAsync(sum,   0, (size_t)C * sizeof(float), stream);
    if (e0 != cudaSuccess) return cuda_last_status();
    cudaError_t e1 = cudaMemsetAsync(sumsq, 0, (size_t)C * sizeof(float), stream);
    if (e1 != cudaSuccess) return cuda_last_status();

    int blocks = (int)(((int64_t)N * C * HW + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;

    cudaGetLastError();
    bn_impl::bn_fwd_stats_f16_atomic<<<blocks, threads, 0, stream>>>(x, sum, sumsq, N, C, HW);
    auto st0 = cuda_last_status();
    if (st0 != Status::Ok) return st0;

    const int blocksC = (C + threads - 1) / threads;
    cudaGetLastError();
    bn_impl::bn_finalize_mean_var<<<blocksC, threads, 0, stream>>>(
        sum, sumsq, C, 1.0f / (float)(N * HW));
    auto st1 = cuda_last_status();
    if (st1 != Status::Ok) return st1;

    cudaGetLastError();
    bn_impl::bn_var_to_rstd_inplace<<<blocksC, threads, 0, stream>>>(sumsq, C, A.eps);
    auto st2 = cuda_last_status();
    if (st2 != Status::Ok) return st2;

    cudaGetLastError();
    bn_impl::bn_fwd_apply_f16<<<blocks, threads, 0, stream>>>(
        x, gamma, beta, sum, sumsq, y, N, C, HW);
    return cuda_last_status();
  }

  // ---------------- inference path ----------------
  {
    int blocks = (int)(((int64_t)N * C * HW + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;

    cudaGetLastError();
    bn_impl::bn_infer_apply_f16<<<blocks, threads, 0, stream>>>(
        x, gamma, beta, running_mean, running_var, y, N, C, HW, A.eps);
    return cuda_last_status();
  }
}

KernelVariant make_batchnorm_fwd_f16_variant() {
  KernelVariant v{};
  v.name = "batchnorm_fwd_f16_nchw";
  v.priority = 10;
  v.flags = 0;
  v.expected_attr_schema_id = kSchema_BNEP;
  v.launch = bn_fwd_f16_launch;
  v.supported = bn_fwd_f16_supported;
  v.query_workspace = bn_fwd_f16_workspace;
  return v;
}

// ============================================================================
// BWD
// ============================================================================

static bool bn_bwd_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void*) {

  if (!inputs || !outputs) return false;

  // affine: inputs 5 => x,dy,gamma,save_mean,save_rstd ; outputs 3 => dx,dgamma,dbeta
  // noaff : inputs 4 => x,dy,save_mean,save_rstd       ; outputs 1 => dx (NotImplemented)
  const bool affine = (num_inputs == 5 && num_outputs == 3);
  const bool noaff  = (num_inputs == 4 && num_outputs == 1);
  if (!(affine || noaff)) return false;

  const auto& X  = inputs[0];
  const auto& dY = inputs[1];
  if (!is_f16_4d(X) || !is_f16_4d(dY)) return false;
  if (!is_contig_nchw_4d(X) || !is_contig_nchw_4d(dY)) return false;
  for (int i = 0; i < 4; ++i) if (dY.shape[i] != X.shape[i]) return false;

  const int64_t C = X.shape[1];

  const auto& SM = inputs[affine ? 3 : 2];
  const auto& SR = inputs[affine ? 4 : 3];
  if (!is_f32_1d(SM) || !is_f32_1d(SR)) return false;
  if (SM.shape[0] != C || SR.shape[0] != C) return false;

  const auto& dX = outputs[0];
  if (!is_f16_4d(dX) || !is_contig_nchw_4d(dX)) return false;
  for (int i = 0; i < 4; ++i) if (dX.shape[i] != X.shape[i]) return false;

  if (affine) {
    const auto& G = inputs[2];
    if (!is_f16_1d(G) || G.shape[0] != C) return false;

    const auto& dG = outputs[1];
    const auto& dB = outputs[2];
    if (!is_f32_1d(dG) || !is_f32_1d(dB)) return false;
    if (dG.shape[0] != C || dB.shape[0] != C) return false;
  }

  return true;
}

static size_t bn_bwd_f16_workspace(const TensorDesc*, int, const void*) { return 0; }

static Status bn_bwd_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void*,
    void*, size_t,
    cudaStream_t stream) {

  if (!bn_bwd_f16_supported(inputs, num_inputs, outputs, num_outputs, nullptr)) {
    return Status::InvalidArgument;
  }

  const bool affine = (num_inputs == 5);
  if (!affine) return Status::NotImplemented;

  const auto& X  = inputs[0];
  const auto& dY = inputs[1];

  const __half* x  = (const __half*)X.data;
  const __half* dy = (const __half*)dY.data;

  const __half* gamma = (const __half*)inputs[2].data;
  const float* mean = (const float*)inputs[3].data;
  const float* rstd = (const float*)inputs[4].data;

  __half* dx = (__half*)outputs[0].data;
  float* dgamma = (float*)outputs[1].data; // sum(dy*xhat)
  float* dbeta  = (float*)outputs[2].data; // sum(dy)

  const int N  = (int)X.shape[0];
  const int C  = (int)X.shape[1];
  const int H  = (int)X.shape[2];
  const int W  = (int)X.shape[3];
  const int HW = H * W;

  std::fprintf(stderr,
    "[BN bwd f16] affine=1 X=%p dY=%p G=%p mean=%p rstd=%p | dX=%p dG=%p dB=%p | N=%d C=%d H=%d W=%d\n",
    (const void*)x, (const void*)dy, (const void*)gamma,
    (const void*)mean, (const void*)rstd,
    (void*)dx, (void*)dgamma, (void*)dbeta,
    N, C, H, W);

  const int threads = 256;
  int blocks = (int)(((int64_t)N * C * HW + threads - 1) / threads);
  if (blocks > 4096) blocks = 4096;

  // Pass A: dbeta=sum(dy) and dgamma=sum(dy*xhat)
  cudaError_t e0 = cudaMemsetAsync(dgamma, 0, (size_t)C * sizeof(float), stream);
  if (e0 != cudaSuccess) return cuda_last_status();
  cudaError_t e1 = cudaMemsetAsync(dbeta,  0, (size_t)C * sizeof(float), stream);
  if (e1 != cudaSuccess) return cuda_last_status();

  cudaGetLastError();
  bn_impl::bn_bwd_sums_f16_atomic<<<blocks, threads, 0, stream>>>(
      x, dy, mean, rstd, /*sum_dy=*/dbeta, /*sum_dy_xhat=*/dgamma, N, C, HW);
  auto st0 = cuda_last_status();
  if (st0 != Status::Ok) return st0;

  // Pass B: dx uses sums + gamma
  cudaGetLastError();
  bn_impl::bn_bwd_dx_f16<<<blocks, threads, 0, stream>>>(
      x, dy, gamma, mean, rstd, /*sum_dy=*/dbeta, /*sum_dy_xhat=*/dgamma, dx, N, C, HW);
  return cuda_last_status();
}

KernelVariant make_batchnorm_bwd_f16_variant() {
  KernelVariant v{};
  v.name = "batchnorm_bwd_f16_nchw_affine";
  v.priority = 10;
  v.flags = 0;
  v.expected_attr_schema_id = 0; // no attrs
  v.launch = bn_bwd_f16_launch;
  v.supported = bn_bwd_f16_supported;
  v.query_workspace = bn_bwd_f16_workspace;
  return v;
}

} // namespace aicf::cuda
