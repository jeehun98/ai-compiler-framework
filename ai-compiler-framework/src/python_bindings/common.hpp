#pragma once
#include <torch/extension.h>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_pack.hpp>

namespace aicf_py {

// -------------------------
// 1) Tensor validation (v0.1)
// - CUDA + contiguous + f16/f32 only
// -------------------------
inline bool is_cuda_contig(const torch::Tensor& t) {
  return t.defined() && t.is_cuda() && t.is_contiguous();
}

inline bool is_cuda_f16_or_f32(const torch::Tensor& t) {
  const auto st = t.scalar_type();
  return st == at::kHalf || st == at::kFloat;
}

inline void check_tensor_v0(const torch::Tensor& t, const char* what) {
  TORCH_CHECK(t.defined(), what, ": undefined tensor");
  TORCH_CHECK(t.is_cuda(), what, ": must be CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), what, ": must be contiguous (v0.1)");
  TORCH_CHECK(is_cuda_f16_or_f32(t), what, ": dtype must be float16 or float32 (v0.1)");
}

// -------------------------
// 2) TensorDesc conversion (v0.1)
// - rank <= kMaxRank
// - stride is treated as contiguous stride by spec
// - contiguous flag is explicitly set true
// -------------------------
inline aicf::cuda::DType to_aicf_dtype(const torch::Tensor& t) {
  if (t.scalar_type() == at::kHalf)  return aicf::cuda::DType::kF16;
  if (t.scalar_type() == at::kFloat) return aicf::cuda::DType::kF32;
  // should be filtered by check_tensor_v0()
  return aicf::cuda::DType::kF32;
}

inline aicf::cuda::TensorDesc to_desc_v0(const torch::Tensor& t) {
  aicf::cuda::TensorDesc d{};
  d.data  = reinterpret_cast<void*>(t.data_ptr());
  d.dtype = to_aicf_dtype(t);

  const int64_t rank64 = t.dim();
  TORCH_CHECK(rank64 >= 0 && rank64 <= aicf::cuda::kMaxRank, "rank too large");

  // TensorDesc uses named union: d.r.rank / d.r.ndim
  d.r.rank = static_cast<int32_t>(rank64);

  for (int i = 0; i < d.r.rank; ++i) {
    d.shape[i] = t.size(i);
  }

  // contiguous stride from shape (spec-fixed)
  int64_t st = 1;
  for (int i = d.r.rank - 1; i >= 0; --i) {
    d.stride[i] = st;
    st *= d.shape[i];
  }

  // v0.1 contract: binding only accepts contiguous tensors
  d.contiguous = true;

  // optional metadata (keep defaults)
  // d.alignment = 0;
  // d.device = 0;

  return d;
}

// -------------------------
// 3) Status handling
// -------------------------
inline bool status_ok(aicf::Status s) {
  return aicf::ok(s);
}

// -------------------------
// 4) Stream policy (v0.1)
// - default: nullptr handle (runtime decides current stream)
// -------------------------
inline aicf::Stream default_stream() {
  aicf::Stream s{};
  s.handle = nullptr;
  return s;
}

} // namespace aicf_py
