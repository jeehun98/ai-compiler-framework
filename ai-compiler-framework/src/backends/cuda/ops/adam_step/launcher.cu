#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

#include <aicf/core/status.hpp>
#include <aicf/backends/cuda/ops/adam_step/api.hpp>

#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_blob.hpp>   // ✅ AttrPack 대신

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

static inline bool any_bad_alias_v2(
    const TensorDesc& P, const TensorDesc& G, const TensorDesc& M, const TensorDesc& V,
    const TensorDesc& Pout, const TensorDesc& Mout, const TensorDesc& Vout) {

  if (same_ptr(G, P) || same_ptr(G, M) || same_ptr(G, V) ||
      same_ptr(G, Pout) || same_ptr(G, Mout) || same_ptr(G, Vout)) {
    return true;
  }

  if (same_ptr(M, V) || same_ptr(Mout, Vout)) return true;

  if (same_ptr(P, M) || same_ptr(P, V) || same_ptr(Pout, Mout) || same_ptr(Pout, Vout) ||
      same_ptr(Pout, M) || same_ptr(Pout, V) || same_ptr(P, Mout) || same_ptr(P, Vout)) {
    return true;
  }

  return false;
}

// ---------------- adam_step_v2 ----------------
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

  // -------- attrs (AttrBlob) --------
  const AttrBlob* ab = attr ? static_cast<const AttrBlob*>(attr) : nullptr;
  if (!ab) return aicf::Status::InvalidArgument;

  // 아래 API는 AttrBlob 설계에 맞춰서 조정하면 됨
  const float lr    = ab->get_f32_or("lr",   NAN);
  const float beta1 = ab->get_f32_or("beta1",NAN);
  const float beta2 = ab->get_f32_or("beta2",NAN);
  const float eps   = ab->get_f32_or("eps",  NAN);

  // 필수 키 검사: NaN이면 missing으로 간주
  if (!(lr == lr && beta1 == beta1 && beta2 == beta2 && eps == eps))
    return aicf::Status::InvalidArgument;

  const int64_t n = numel(P);
  if (n <= 0) return aicf::Status::Ok;

  const int threads = 256;
  const int blocks  = (int)((n + threads - 1) / threads);

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

} // namespace aicf::cuda
