#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

#include <aicf/backends/cuda/ops/adam_step/api.hpp>

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
// AttrBlob schema: AdamStep
// payload: float32 lr, beta1, beta2, eps  (16 bytes)
// schema_id: 'ADAM' (0x4D414441 little-endian)
// schema_id==0 allowed -> defaults
// -------------------------
static constexpr uint32_t kSchema_ADAM = 0x4D414441u;

static inline float read_f32_le(const uint8_t* p) {
  float v;
  std::memcpy(&v, p, sizeof(float));
  return v;
}

struct AdamAttrs {
  float lr;
  float beta1;
  float beta2;
  float eps;
};

static inline AdamAttrs get_adam_attrs_from_attr(const AttrBlob& a) {
  AdamAttrs out{1e-3f, 0.9f, 0.999f, 1e-8f};

  if (a.schema_id == 0) return out;
  if (a.schema_id != kSchema_ADAM) return out;
  if (!a.data || a.bytes < 16) return out;

  const auto* p = static_cast<const uint8_t*>(a.data);
  out.lr    = read_f32_le(p + 0);
  out.beta1 = read_f32_le(p + 4);
  out.beta2 = read_f32_le(p + 8);
  out.eps   = read_f32_le(p + 12);
  return out;
}

// -------------------------
// helpers
// -------------------------
static inline int64_t numel(const TensorDesc& d) {
  int64_t n = 1;
  const int r = d.rank();
  for (int i = 0; i < r; ++i) n *= (int64_t)d.shape[i];
  return n;
}

static inline bool same_shape(const TensorDesc& a, const TensorDesc& b) {
  if (a.rank() != b.rank()) return false;
  for (int i = 0; i < a.rank(); ++i) {
    if (a.shape[i] != b.shape[i]) return false;
  }
  return true;
}

static inline bool is_contig_anyrank(const TensorDesc& d) {
  return d.contiguous && (d.rank() >= 1);
}

static inline bool is_f32_contig_anyrank(const TensorDesc& d) {
  return (d.dtype == DType::kF32) && is_contig_anyrank(d);
}

// BC1/BC2: rank0 scalar float32 contiguous
static inline bool is_scalar_f32_contig(const TensorDesc& d) {
  return (d.dtype == DType::kF32) && (d.rank() == 0) && d.contiguous;
}

static inline bool ptr_eq(const TensorDesc& a, const TensorDesc& b) { return a.data == b.data; }

// alias policy (migration-only, conservative):
// - allow Pout==P, Mout==M, Vout==V (in-place OK)
// - forbid ANY other aliasing, especially G with anything, and cross-alias between P/M/V groups
static inline bool any_bad_alias(
    const TensorDesc& P, const TensorDesc& G, const TensorDesc& M, const TensorDesc& V,
    const TensorDesc& Pout, const TensorDesc& Mout, const TensorDesc& Vout) {

  // G must not alias anything
  if (ptr_eq(G, P) || ptr_eq(G, M) || ptr_eq(G, V) ||
      ptr_eq(G, Pout) || ptr_eq(G, Mout) || ptr_eq(G, Vout)) {
    return true;
  }

  // allow exact in-place per-buffer
  const bool p_inplace = ptr_eq(Pout, P);
  const bool m_inplace = ptr_eq(Mout, M);
  const bool v_inplace = ptr_eq(Vout, V);

  if (!p_inplace) {
    if (ptr_eq(Pout, M) || ptr_eq(Pout, V) || ptr_eq(Pout, Mout) || ptr_eq(Pout, Vout)) return true;
  }
  if (!m_inplace) {
    if (ptr_eq(Mout, P) || ptr_eq(Mout, V) || ptr_eq(Mout, Pout) || ptr_eq(Mout, Vout)) return true;
  }
  if (!v_inplace) {
    if (ptr_eq(Vout, P) || ptr_eq(Vout, M) || ptr_eq(Vout, Pout) || ptr_eq(Vout, Mout)) return true;
  }

  // forbid base state alias (M==V etc.)
  if (ptr_eq(M, V)) return true;
  if (ptr_eq(P, M) || ptr_eq(P, V)) return true;

  return false;
}

static inline int choose_blocks_1d(int64_t n, int threads) {
  int64_t b = (n + threads - 1) / threads;
  if (b < 1) b = 1;
  const int64_t kMaxBlocks = 65535;
  if (b > kMaxBlocks) b = kMaxBlocks;
  return (int)b;
}

static size_t adam_step_workspace(const TensorDesc*, int, const void*) { return 0; }

// -------------------------
// checks
// -------------------------
static inline bool adam_step_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 6 || num_outputs != 3) return false;

  const TensorDesc& P   = inputs[0];
  const TensorDesc& G   = inputs[1];
  const TensorDesc& M   = inputs[2];
  const TensorDesc& V   = inputs[3];
  const TensorDesc& BC1 = inputs[4];
  const TensorDesc& BC2 = inputs[5];

  const TensorDesc& Pout = outputs[0];
  const TensorDesc& Mout = outputs[1];
  const TensorDesc& Vout = outputs[2];

  if (!is_f32_contig_anyrank(P) || !is_f32_contig_anyrank(G) ||
      !is_f32_contig_anyrank(M) || !is_f32_contig_anyrank(V) ||
      !is_f32_contig_anyrank(Pout) || !is_f32_contig_anyrank(Mout) || !is_f32_contig_anyrank(Vout)) {
    return false;
  }

  if (!is_scalar_f32_contig(BC1) || !is_scalar_f32_contig(BC2)) return false;

  if (!same_shape(P, G) || !same_shape(P, M) || !same_shape(P, V)) return false;
  if (!same_shape(P, Pout) || !same_shape(M, Mout) || !same_shape(V, Vout)) return false;

  if (any_bad_alias(P, G, M, V, Pout, Mout, Vout)) return false;

  const int64_t n = numel(P);
  return (n >= 0);
}

// -------------------------
// Variant: F32
// -------------------------
static bool adam_step_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return adam_step_check(inputs, num_inputs, outputs, num_outputs);
}

static Status adam_step_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!adam_step_check(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }
  if (!attr) return Status::InvalidArgument;

  const TensorDesc& P   = inputs[0];
  const TensorDesc& G   = inputs[1];
  const TensorDesc& M   = inputs[2];
  const TensorDesc& V   = inputs[3];
  const TensorDesc& BC1 = inputs[4];
  const TensorDesc& BC2 = inputs[5];

  TensorDesc& Pout = outputs[0];
  TensorDesc& Mout = outputs[1];
  TensorDesc& Vout = outputs[2];

  const int64_t n = numel(P);
  if (n <= 0) return Status::Ok;

  const AttrBlob& ab = *static_cast<const AttrBlob*>(attr);
  const AdamAttrs a = get_adam_attrs_from_attr(ab);

  // ✅ oop 안전 보장:
  // kernel은 Pout을 "기존 값에서 - lr*..." 형태로 업데이트하므로,
  // Pout!=P 인 경우 P 값을 Pout에 먼저 복사한다.
  if (!ptr_eq(Pout, P)) {
    const size_t bytes = (size_t)n * sizeof(float);
    cudaError_t e = cudaMemcpyAsync(Pout.data, P.data, bytes, cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return cuda_to_status(e);
  }

  constexpr int kThreads = 256;
  const int blocks = choose_blocks_1d(n, kThreads);

  cudaGetLastError(); // clear
  adam_step_f32_kernel_v2<<<blocks, kThreads, 0, stream>>>(
      (float*)Pout.data,
      (const float*)G.data,
      (const float*)M.data,
      (const float*)V.data,
      (float*)Mout.data,
      (float*)Vout.data,
      n,
      a.lr, a.beta1, a.beta2, a.eps,
      (const float*)BC1.data,
      (const float*)BC2.data);

  return cuda_last_status();
}

KernelVariant make_adam_step_f32_variant() {
  KernelVariant v{};
  v.name = "adam_step_f32_v2";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = kSchema_ADAM; // exec에서 ADAM으로 보내야 매칭됨
  v.launch = adam_step_f32_launch;
  v.supported = adam_step_f32_supported;
  v.query_workspace = adam_step_workspace;
  return v;
}

__global__ void adam_step_f32_kernel_v2(
    float* __restrict__ Pout,
    const float* __restrict__ G,
    const float* __restrict__ M,
    const float* __restrict__ V,
    float* __restrict__ Mout,
    float* __restrict__ Vout,
    int64_t n,
    float lr, float beta1, float beta2, float eps,
    const float* __restrict__ bc1,
    const float* __restrict__ bc2) {

  const float bc1v = bc1 ? bc1[0] : 1.0f;
  const float bc2v = bc2 ? bc2[0] : 1.0f;

  for (int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
       i < n;
       i += (int64_t)blockDim.x * (int64_t)gridDim.x) {

    const float g = G[i];

    const float m_new = beta1 * M[i] + (1.0f - beta1) * g;
    const float v_new = beta2 * V[i] + (1.0f - beta2) * (g * g);

    Mout[i] = m_new;
    Vout[i] = v_new;

    const float m_hat = m_new / bc1v;
    const float v_hat = v_new / bc2v;

    const float denom = sqrtf(v_hat) + eps;

    // ✅ launcher가 oop일 때 P->Pout copy를 보장하므로,
    // 여기서는 "Pout = Pout - ..." 방식이 항상 안전해짐.
    Pout[i] = Pout[i] - lr * (m_hat / denom);
  }
}

} // namespace aicf::cuda
