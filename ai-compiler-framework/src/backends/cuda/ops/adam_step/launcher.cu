#include <aicf/backends/cuda/ops/adam_step/api.hpp>

#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_pack.hpp>

#include <cuda_runtime.h>
#include <cstring>   // strcmp
#include <cstdint>

#include "kernels.cuh"

namespace aicf::cuda {

static inline int64_t numel(const TensorDesc& d) {
  int64_t n = 1;
  for (int i = 0; i < d.r.rank; ++i) n *= (int64_t)d.shape[i];
  return n;
}

static inline bool same_shape(const TensorDesc& a, const TensorDesc& b) {
  if (a.r.rank != b.r.rank) return false;
  for (int i = 0; i < a.r.rank; ++i)
    if (a.shape[i] != b.shape[i]) return false;
  return true;
}

static inline bool is_contig(const TensorDesc& d) { return d.contiguous; }

static inline bool is_scalar_f32_contig(const TensorDesc& d) {
  return d.dtype == DType::kF32 && d.r.rank == 0 && is_contig(d);
}

static inline bool same_ptr(const TensorDesc& a, const TensorDesc& b) {
  return a.data == b.data;
}

// v2에서는 out-of-place 허용하므로 alias 규칙을 "필수 금지"만 강하게 잡는다.
// - G는 어떤 것과도 alias 금지
// - M과 V는 서로 alias 금지 (상태 분리)
// - (선택) P와 M/V가 alias면 금지 (보통 금지하는 게 안전)
static inline bool any_bad_alias_v2(
    const TensorDesc& P, const TensorDesc& G, const TensorDesc& M, const TensorDesc& V,
    const TensorDesc& Pout, const TensorDesc& Mout, const TensorDesc& Vout) {

  // forbid G aliasing anything (inputs/outputs)
  if (same_ptr(G, P) || same_ptr(G, M) || same_ptr(G, V) ||
      same_ptr(G, Pout) || same_ptr(G, Mout) || same_ptr(G, Vout)) {
    return true;
  }

  // state must be distinct
  if (same_ptr(M, V) || same_ptr(Mout, Vout)) return true;

  // forbid P with state overlap (strong safety)
  if (same_ptr(P, M) || same_ptr(P, V) || same_ptr(Pout, Mout) || same_ptr(Pout, Vout) ||
      same_ptr(Pout, M) || same_ptr(Pout, V) || same_ptr(P, Mout) || same_ptr(P, Vout)) {
    return true;
  }

  return false;
}

// ---------------- adam_step_v2 ----------------
// inputs:  P, G, M, V, bc1_inv, bc2_inv (device scalar tensors)
// outputs: P, M, V  (but now Mout/Vout may be different ptrs than Min/Vin)
aicf::Status adam_step_v2(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (num_inputs != 6 || num_outputs != 3) return aicf::Status::InvalidArgument;

  const TensorDesc& P   = inputs[0];
  const TensorDesc& G   = inputs[1];
  const TensorDesc& M   = inputs[2];
  const TensorDesc& V   = inputs[3];
  const TensorDesc& BC1 = inputs[4];
  const TensorDesc& BC2 = inputs[5];

  TensorDesc& Pout = outputs[0];
  TensorDesc& Mout = outputs[1];
  TensorDesc& Vout = outputs[2];

  // f32 only
  if (P.dtype != DType::kF32 || G.dtype != DType::kF32 ||
      M.dtype != DType::kF32 || V.dtype != DType::kF32)
    return aicf::Status::NotImplemented;

  if (!is_scalar_f32_contig(BC1) || !is_scalar_f32_contig(BC2))
    return aicf::Status::InvalidArgument;

  if (!same_shape(P, G) || !same_shape(P, M) || !same_shape(P, V))
    return aicf::Status::InvalidArgument;

  if (!same_shape(P, Pout) || !same_shape(M, Mout) || !same_shape(V, Vout))
    return aicf::Status::InvalidArgument;

  if (!is_contig(P) || !is_contig(G) || !is_contig(M) || !is_contig(V) ||
      !is_contig(Pout) || !is_contig(Mout) || !is_contig(Vout))
    return aicf::Status::NotImplemented;

  if (any_bad_alias_v2(P, G, M, V, Pout, Mout, Vout))
    return aicf::Status::InvalidArgument;

  const AttrPack* pack = (const AttrPack*)attr;
  if (!pack) return aicf::Status::InvalidArgument;

  float lr=0, beta1=0, beta2=0, eps=0;
  bool has_lr=false, has_b1=false, has_b2=false, has_eps=false;

  for (int i = 0; i < pack->size; ++i) {
    const auto& kv = pack->items[i];
    if (!std::strcmp(kv.key, "lr"))         { lr = kv.val.f32; has_lr=true; }
    else if (!std::strcmp(kv.key, "beta1")) { beta1 = kv.val.f32; has_b1=true; }
    else if (!std::strcmp(kv.key, "beta2")) { beta2 = kv.val.f32; has_b2=true; }
    else if (!std::strcmp(kv.key, "eps"))   { eps = kv.val.f32; has_eps=true; }
  }
  if (!(has_lr && has_b1 && has_b2 && has_eps))
    return aicf::Status::InvalidArgument;

  const int64_t n = numel(P);
  if (n <= 0) return aicf::Status::Ok;

  const int threads = 256;
  const int blocks  = (int)((n + threads - 1) / threads);

  // true out-of-place support:
  //  - reads Min/Vin from inputs[2]/inputs[3]
  //  - writes Mout/Vout to outputs[1]/outputs[2]
  adam_step_f32_kernel_v2<<<blocks, threads, 0, stream>>>(
      (float*)Pout.data,
      (const float*)G.data,
      (const float*)M.data,
      (const float*)V.data,
      (float*)Mout.data,
      (float*)Vout.data,
      n, lr, beta1, beta2, eps,
      (const float*)BC1.data,
      (const float*)BC2.data);

  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) return aicf::Status::Internal;

  return aicf::Status::Ok;
}

// ---------------- KernelVariant (v2) ----------------
static bool supported_adam_step_f32_v2(
    const TensorDesc* in, int ni,
    const TensorDesc* out, int no,
    const void* /*attr*/) {

  if (ni != 6 || no != 3) return false;

  // P,G,M,V contig f32
  for (int i = 0; i < 4; ++i) {
    if (in[i].dtype != DType::kF32) return false;
    if (!is_contig(in[i])) return false;
  }

  if (!same_shape(in[0], in[1]) || !same_shape(in[0], in[2]) || !same_shape(in[0], in[3]))
    return false;

  if (!is_scalar_f32_contig(in[4])) return false;
  if (!is_scalar_f32_contig(in[5])) return false;

  // outputs contig f32, same shapes
  for (int i = 0; i < 3; ++i) {
    if (out[i].dtype != DType::kF32) return false;
    if (!is_contig(out[i])) return false;
  }
  if (!same_shape(out[0], in[0]) || !same_shape(out[1], in[2]) || !same_shape(out[2], in[3]))
    return false;

  if (any_bad_alias_v2(in[0], in[1], in[2], in[3], out[0], out[1], out[2]))
    return false;

  return true;
}

static size_t query_ws_adam_step(const TensorDesc*, int, const void*) { return 0; }

static aicf::Status launch_adam_step_v2(
    const TensorDesc* inputs, int ni,
    TensorDesc* outputs, int no,
    const void* attr,
    void* ws, size_t ws_bytes,
    cudaStream_t stream) {
  return adam_step_v2(inputs, ni, outputs, no, attr, ws, ws_bytes, stream);
}

KernelVariant make_adam_step_f32_variant() {
  KernelVariant kv{};

  // v2를 기본으로 올려버리는 게 네 IRExecutor SAFE 디버그에 유리함
  kv.name = "adam_step_f32_v2_true_oop";
  kv.priority = 0;
  kv.query_workspace = query_ws_adam_step;
  kv.supported = supported_adam_step_f32_v2;
  kv.launch = launch_adam_step_v2;

  return kv;
}

} // namespace aicf::cuda
