// ============================================================================
// src/backends/cuda/ops/cross_entropy_loss/launcher.cu
// - logits:  [N,C] f32 contig
// - targets: [N]   int32 contig
// - outputs(fwd): loss[1] f32   (mean or sum)
// - inputs(bwd): logits, targets, grad_loss[1] (f32)
// - outputs(bwd): dlogits [N,C] f32
//
// AttrBlob (schema_id = 'XENT'):
//   payload: int32 ignore_index, int32 reduction (0 mean, 1 sum)
// ============================================================================

#include <cuda_runtime.h>
#include <math_constants.h> // CUDART_INF_F

#include <cstdint>
#include <cinttypes>
#include <climits>

#include <aicf/backends/cuda/registry/status.hpp>
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_blob.hpp>

#include "kernels.cuh"

namespace aicf::cuda {

// ---- cuda error -> Status ----
static inline Status cuda_to_status(cudaError_t e) {
  return (e == cudaSuccess) ? Status::Ok : Status::Internal;
}
static inline Status cuda_last_status() {
  return cuda_to_status(cudaGetLastError());
}

// -------------------------
// helpers
// -------------------------
namespace {

static inline bool is_contig_anyrank(const TensorDesc& T) {
  const int r = T.rank();
  if (r <= 0) return false;
  if (T.stride[r - 1] != 1) return false;
  for (int i = r - 2; i >= 0; --i) {
    if (T.stride[i] != T.shape[i + 1] * T.stride[i + 1]) return false;
  }
  return true;
}

static inline bool same_shape_anyrank(const TensorDesc& A, const TensorDesc& B) {
  if (A.rank() != B.rank()) return false;
  for (int i = 0; i < A.rank(); ++i) {
    if (A.shape[i] != B.shape[i]) return false;
  }
  return true;
}

static size_t xent_workspace(const TensorDesc*, int, const void*) { return 0; }

struct XentAttr {
  int ignore_index;
  int reduction; // 0 mean, 1 sum
};

static constexpr uint32_t kSchemaXENT =
    ((uint32_t)('X')) | ((uint32_t)('E') << 8) | ((uint32_t)('N') << 16) | ((uint32_t)('T') << 24);

// ✅ attr points to AttrBlob (not raw payload)
static inline XentAttr parse_attr(const void* attr, uint32_t expected_schema_id) {
  XentAttr a{};
  a.ignore_index = -100;
  a.reduction = 0; // mean default

  if (!attr) return a;

  const AttrBlob* b = (const AttrBlob*)attr;

  // schema_id==0: unspecified -> allow default/payload if present
  if (expected_schema_id != 0 && b->schema_id != 0 && b->schema_id != expected_schema_id) {
    return a;
  }

  if (!b->data || b->bytes < 8) return a;

  const int32_t* p = (const int32_t*)b->data;
  a.ignore_index = (int)p[0];
  a.reduction    = (int)p[1];

  // clamp
  a.reduction = (a.reduction == 0) ? 0 : 1;
  return a;
}

// logits [N,C], targets [N], loss [1]
static inline bool xent_fwd_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2) return false;
  if (num_outputs != 1) return false;

  const TensorDesc& L = inputs[0];
  const TensorDesc& T = inputs[1];
  const TensorDesc& O = outputs[0];

  if (L.dtype != DType::kF32) return false;
  if (T.dtype != DType::kI32) return false;
  if (O.dtype != DType::kF32) return false;

  if (!is_contig_anyrank(L) || !is_contig_anyrank(T) || !is_contig_anyrank(O)) return false;

  if (L.rank() != 2) return false;
  const int64_t N = L.shape[0];
  const int64_t C = L.shape[1];
  if (N <= 0 || C <= 0) return false;
  if (N > (int64_t)INT_MAX || C > (int64_t)INT_MAX) return false;

  if (T.rank() != 1) return false;
  if (T.shape[0] != N) return false;

  if (O.rank() != 1 || O.shape[0] != 1) return false;

  return true;
}

// inputs(bwd): logits, targets, grad_loss[1]
// outputs(bwd): dlogits
static inline bool xent_bwd_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 3) return false;
  if (num_outputs != 1) return false;

  const TensorDesc& L  = inputs[0];
  const TensorDesc& T  = inputs[1];
  const TensorDesc& G  = inputs[2];
  const TensorDesc& dL = outputs[0];

  if (L.dtype != DType::kF32) return false;
  if (T.dtype != DType::kI32) return false;
  if (G.dtype != DType::kF32) return false;
  if (dL.dtype != DType::kF32) return false;

  if (!is_contig_anyrank(L) || !is_contig_anyrank(T) || !is_contig_anyrank(G) || !is_contig_anyrank(dL))
    return false;

  if (L.rank() != 2) return false;
  if (!same_shape_anyrank(L, dL)) return false;

  const int64_t N = L.shape[0];
  if (T.rank() != 1 || T.shape[0] != N) return false;

  if (G.rank() != 1 || G.shape[0] != 1) return false;

  if (N > (int64_t)INT_MAX || L.shape[1] > (int64_t)INT_MAX) return false;

  return true;
}

} // anonymous namespace

// ============================================================================
// Forward variant
// ============================================================================
static bool xent_fwd_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void*) {
  return xent_fwd_check(inputs, num_inputs, outputs, num_outputs);
}

static Status xent_fwd_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!xent_fwd_check(inputs, num_inputs, outputs, num_outputs))
    return Status::InvalidArgument;

  const TensorDesc& L = inputs[0];
  const TensorDesc& T = inputs[1];
  TensorDesc& O = outputs[0];

  const int N = (int)L.shape[0];
  const int C = (int)L.shape[1];

  const XentAttr a = parse_attr(attr, kSchemaXENT);

  float* d_loss = nullptr;
  int* d_valid = nullptr;

  cudaError_t e1 = cudaMallocAsync(&d_loss, sizeof(float), stream);
  cudaError_t e2 = cudaMallocAsync(&d_valid, sizeof(int), stream);
  if (e1 != cudaSuccess || e2 != cudaSuccess) return Status::Internal;

  cudaMemsetAsync(d_loss, 0, sizeof(float), stream);
  cudaMemsetAsync(d_valid, 0, sizeof(int), stream);

  constexpr int kThreads = 256;
  const int blocks = N;

  cudaGetLastError();
  xent_impl::xent_fwd_sum_f32<<<blocks, kThreads, 0, stream>>>(
      (const float*)L.data,
      (const int32_t*)T.data,
      N, C,
      a.ignore_index,
      d_loss, d_valid);

  Status st = cuda_last_status();
  if (st != Status::Ok) {
    cudaFreeAsync(d_loss, stream);
    cudaFreeAsync(d_valid, stream);
    return st;
  }

  xent_impl::xent_finalize_loss_f32<<<1, 32, 0, stream>>>(
      d_loss, d_valid, a.reduction, (float*)O.data);

  st = cuda_last_status();

  cudaFreeAsync(d_loss, stream);
  cudaFreeAsync(d_valid, stream);
  return st;
}

KernelVariant make_cross_entropy_loss_fwd_f32_variant() {
  KernelVariant v{};
  v.name = "cross_entropy_loss_fwd_f32";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = kSchemaXENT;
  v.launch = xent_fwd_launch;
  v.supported = xent_fwd_supported;
  v.query_workspace = xent_workspace;
  return v;
}

// ============================================================================
// Backward variant
// ============================================================================
static bool xent_bwd_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void*) {
  return xent_bwd_check(inputs, num_inputs, outputs, num_outputs);
}

static Status xent_bwd_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!xent_bwd_check(inputs, num_inputs, outputs, num_outputs))
    return Status::InvalidArgument;

  const TensorDesc& L = inputs[0];
  const TensorDesc& T = inputs[1];
  const TensorDesc& G = inputs[2]; // grad_loss device scalar
  TensorDesc& dL = outputs[0];

  const int N = (int)L.shape[0];
  const int C = (int)L.shape[1];

  const XentAttr a = parse_attr(attr, kSchemaXENT);

  int* d_valid = nullptr;
  cudaError_t e = cudaMallocAsync(&d_valid, sizeof(int), stream);
  if (e != cudaSuccess) return Status::Internal;
  cudaMemsetAsync(d_valid, 0, sizeof(int), stream);

  if (a.reduction == 0) {
    constexpr int kThreads = 256;
    const int blocks = (N + kThreads - 1) / kThreads;

    cudaGetLastError();
    xent_impl::xent_count_valid_i32<<<blocks, kThreads, 0, stream>>>(
        (const int32_t*)T.data, N, a.ignore_index, d_valid);

    Status st = cuda_last_status();
    if (st != Status::Ok) {
      cudaFreeAsync(d_valid, stream);
      return st;
    }
  } else {
    // sum mode: denom = 1 (to reuse same kernel)
    xent_impl::xent_set_int_scalar<<<1, 32, 0, stream>>>(d_valid, 1);
    Status st = cuda_last_status();
    if (st != Status::Ok) {
      cudaFreeAsync(d_valid, stream);
      return st;
    }
  }

  constexpr int kThreads2 = 256;
  const int blocks2 = N;

  cudaGetLastError();
  xent_impl::xent_bwd_f32<<<blocks2, kThreads2, 0, stream>>>(
      (const float*)L.data,
      (const int32_t*)T.data,
      N, C,
      a.ignore_index,
      (const float*)G.data,    // device scalar ptr
      (const int*)d_valid,     // device scalar ptr
      a.reduction,
      (float*)dL.data);

  Status st2 = cuda_last_status();
  cudaFreeAsync(d_valid, stream);
  return st2;
}

KernelVariant make_cross_entropy_loss_bwd_f32_variant() {
  KernelVariant v{};
  v.name = "cross_entropy_loss_bwd_f32";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = kSchemaXENT;
  v.launch = xent_bwd_launch;
  v.supported = xent_bwd_supported;
  v.query_workspace = xent_workspace;
  return v;
}

} // namespace aicf::cuda

// ============================================================================
// kernel definitions (same TU, relu-style)
// ============================================================================

#include <math.h>

namespace aicf::cuda::xent_impl {

// ---- small utils ----
__global__ void xent_set_int_scalar(int* p, int v) {
  if (threadIdx.x == 0) p[0] = v;
}

__global__ void xent_finalize_loss_f32(
    const float* __restrict__ loss_sum,
    const int* __restrict__ valid,
    int reduction,
    float* __restrict__ out_loss)
{
  if (threadIdx.x == 0) {
    float v = loss_sum[0];
    int n = valid[0];
    if (reduction == 0) out_loss[0] = (n > 0) ? (v / (float)n) : 0.0f;
    else                out_loss[0] = v;
  }
}

__global__ void xent_count_valid_i32(
    const int32_t* __restrict__ t,
    int N,
    int ignore_index,
    int* __restrict__ out_valid)
{
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N) {
    if (t[i] != ignore_index) atomicAdd(out_valid, 1);
  }
}

// ---- reductions ----
__inline__ __device__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}
__inline__ __device__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
  return v;
}

// ✅ broadcasted block reductions
__inline__ __device__ float block_sum(float v) {
  __shared__ float sh[32];
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;

  v = warp_sum(v);
  if (lane == 0) sh[wid] = v;
  __syncthreads();

  float out = 0.0f;
  const int num_warps = (blockDim.x + 31) >> 5;
  if (wid == 0) {
    out = (lane < num_warps) ? sh[lane] : 0.0f;
    out = warp_sum(out);
    if (lane == 0) sh[0] = out;
  }
  __syncthreads();
  return sh[0];
}

__inline__ __device__ float block_max(float v) {
  __shared__ float sh[32];
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;

  v = warp_max(v);
  if (lane == 0) sh[wid] = v;
  __syncthreads();

  float out = -CUDART_INF_F;
  const int num_warps = (blockDim.x + 31) >> 5;
  if (wid == 0) {
    out = (lane < num_warps) ? sh[lane] : -CUDART_INF_F;
    out = warp_max(out);
    if (lane == 0) sh[0] = out;
  }
  __syncthreads();
  return sh[0];
}

// ---- fwd ----
__global__ void xent_fwd_sum_f32(
    const float* __restrict__ logits,
    const int32_t* __restrict__ targets,
    int N, int C,
    int ignore_index,
    float* __restrict__ out_loss,
    int* __restrict__ out_valid)
{
  int n = (int)blockIdx.x;
  if (n >= N) return;

  int t = targets[n];
  if (t == ignore_index) return;

  // OOB guard
  if ((unsigned)t >= (unsigned)C) return;

  const float* row = logits + (size_t)n * C;

  float m = -CUDART_INF_F;
  for (int c = threadIdx.x; c < C; c += blockDim.x) m = fmaxf(m, row[c]);
  m = block_max(m);

  float s = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) s += __expf(row[c] - m);
  s = block_sum(s);

  if (!(s > 0.0f) || !isfinite(s)) return;

  float logZ = logf(s) + m;

  float zt = row[t];
  if (!isfinite(zt) || !isfinite(logZ)) return;

  float loss_n = logZ - zt;

  if (threadIdx.x == 0) {
    atomicAdd(out_loss, loss_n);
    atomicAdd(out_valid, 1);
  }
}

// ---- bwd ----
__global__ void xent_bwd_f32(
    const float* __restrict__ logits,
    const int32_t* __restrict__ targets,
    int N, int C,
    int ignore_index,
    const float* __restrict__ grad_loss,
    const int* __restrict__ valid,
    int reduction,
    float* __restrict__ dlogits)
{
  int n = (int)blockIdx.x;
  if (n >= N) return;

  int t = targets[n];
  float* dx = dlogits + (size_t)n * C;

  if (t == ignore_index || (unsigned)t >= (unsigned)C) {
    for (int c = threadIdx.x; c < C; c += blockDim.x) dx[c] = 0.0f;
    return;
  }

  const float* row = logits + (size_t)n * C;

  float m = -CUDART_INF_F;
  for (int c = threadIdx.x; c < C; c += blockDim.x) m = fmaxf(m, row[c]);
  m = block_max(m);

  float s = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) s += __expf(row[c] - m);
  s = block_sum(s);

  if (!(s > 0.0f) || !isfinite(s)) {
    for (int c = threadIdx.x; c < C; c += blockDim.x) dx[c] = 0.0f;
    return;
  }

  float g = grad_loss[0];
  if (!isfinite(g)) {
    for (int c = threadIdx.x; c < C; c += blockDim.x) dx[c] = 0.0f;
    return;
  }

  float inv_denom = 1.0f;
  if (reduction == 0) {
    int vc = valid[0];
    inv_denom = (vc > 0) ? (1.0f / (float)vc) : 0.0f;
  }
  float scale = g * inv_denom;

  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float p = __expf(row[c] - m) / s;
    float y = (c == t) ? 1.0f : 0.0f;
    dx[c] = (p - y) * scale;
  }
}

} // namespace aicf::cuda::xent_impl
