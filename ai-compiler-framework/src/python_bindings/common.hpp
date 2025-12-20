#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// NOTE: stream API (portable enough)
#include <ATen/cuda/CUDAContext.h>

#include <aicf/core/status.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>

#include <cstdint>
#include <sstream>
#include <string>

namespace aicf_py {

// -------------------------
// helpers (debug string)
// -------------------------
inline const char* dtype_name(const torch::Tensor& t) {
  switch (t.scalar_type()) {
    case at::kHalf:  return "float16";
    case at::kFloat: return "float32";
    default:         return "other";
  }
}

inline std::string tensor_brief(const torch::Tensor& t) {
  std::ostringstream oss;
  oss << "defined=" << (t.defined() ? "true" : "false");
  if (!t.defined()) return oss.str();

  oss << " device=" << (t.is_cuda() ? "cuda" : "cpu");
  oss << " dtype=" << dtype_name(t);
  oss << " contig=" << (t.is_contiguous() ? "true" : "false");
  oss << " rank=" << t.dim();
  oss << " shape=[";
  for (int i = 0; i < t.dim(); ++i) {
    oss << t.size(i);
    if (i + 1 < t.dim()) oss << ",";
  }
  oss << "]";
  return oss.str();
}

// -------------------------
// v0.2 Tensor validation
// -------------------------
inline void check_tensor_v0_2(const torch::Tensor& t, const char* what) {
  TORCH_CHECK(t.defined(), what, ": undefined tensor");
  TORCH_CHECK(t.is_cuda(), what, ": must be CUDA tensor. got: ", tensor_brief(t));
  TORCH_CHECK(t.is_contiguous(),
              what, ": must be contiguous (binding v0.2). got: ", tensor_brief(t));

  const auto st = t.scalar_type();
  TORCH_CHECK(st == at::kHalf || st == at::kFloat,
              what, ": dtype must be float16 or float32 (binding v0.2). got: ", tensor_brief(t));

  const int64_t rank64 = t.dim();
  TORCH_CHECK(rank64 >= 0 && rank64 <= aicf::cuda::kMaxRank,
              what, ": rank out of range. got rank=", rank64,
              " (kMaxRank=", aicf::cuda::kMaxRank, ")");
}

inline aicf::cuda::DType to_aicf_dtype_strict(const torch::Tensor& t) {
  if (t.scalar_type() == at::kHalf)  return aicf::cuda::DType::kF16;
  if (t.scalar_type() == at::kFloat) return aicf::cuda::DType::kF32;
  TORCH_CHECK(false, "unsupported dtype. got: ", tensor_brief(t));
}

// -------------------------
// v0.2 TensorDesc conversion (contiguous contract)
// -------------------------
inline aicf::cuda::TensorDesc to_contig_desc_v0_2(const torch::Tensor& t) {
  check_tensor_v0_2(t, "to_contig_desc_v0_2(t)");

  aicf::cuda::TensorDesc d{};
  d.data  = const_cast<void*>(t.data_ptr());
  d.dtype = to_aicf_dtype_strict(t);

  const int32_t r = static_cast<int32_t>(t.dim());
  d.r.rank = r;

  for (int i = 0; i < r; ++i) {
    d.shape[i] = t.size(i);
  }

  // contiguous stride derived from shape
  int64_t st = 1;
  for (int i = r - 1; i >= 0; --i) {
    d.stride[i] = st;
    st *= d.shape[i];
  }

  d.contiguous = true;
  return d;
}

// -------------------------
// Stream policy
// -------------------------
inline cudaStream_t current_cuda_stream() {
  // "default" stream is always available. If you later confirm
  // an ATen "current stream" API in your version, switch here.
  return at::cuda::getDefaultCUDAStream().stream();
}

} // namespace aicf_py
