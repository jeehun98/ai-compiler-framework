#pragma once

#include <cstdint>

#include "aicf/backends/cuda/registry/tensor_desc.hpp"

namespace aicf::cuda::shim {

// ============================================================
// v0.2 validate rules
// - DO NOT depend on TensorDesc::ndim() method in shared headers.
// - Use a single canonical rank accessor.
// - Keep checks cheap; binding already enforces CUDA+contig policy.
// ============================================================

// -------- rank / ndim (canonical) --------
// Prefer using the struct field, not a method, to avoid "bound function pointer" accidents.
inline int32_t rank(const TensorDesc& t) {
  // Canonical source of truth:
  // - Binding sets d.r.rank
  return t.r.rank;
}

inline bool is_rank(const TensorDesc& t, int32_t n) {
  return rank(t) == n;
}

// -------- basics --------
inline bool is_contig(const TensorDesc& t) { return t.contiguous; }

// NOTE: align enum names with your actual DType definition.
// If your enum is DType::kF32, change here accordingly.
inline bool is_f32(const TensorDesc& t) { return t.dtype == DType::F32; }
inline bool is_f16(const TensorDesc& t) { return t.dtype == DType::F16; }
inline bool is_bf16(const TensorDesc& t) { return t.dtype == DType::BF16; }

// -------- presets (Plan A / v0.x) --------
inline bool is_f32_contig_1d(const TensorDesc& t) {
  return is_f32(t) && is_contig(t) && is_rank(t, 1);
}

inline bool is_f32_contig_2d(const TensorDesc& t) {
  return is_f32(t) && is_contig(t) && is_rank(t, 2);
}

inline bool is_f16_contig_1d(const TensorDesc& t) {
  return is_f16(t) && is_contig(t) && is_rank(t, 1);
}

inline bool is_f16_contig_2d(const TensorDesc& t) {
  return is_f16(t) && is_contig(t) && is_rank(t, 2);
}

// -------- shape helpers --------
inline bool same_shape_1d(const TensorDesc& a, const TensorDesc& b) {
  return is_rank(a, 1) && is_rank(b, 1) && a.shape[0] == b.shape[0];
}

inline bool same_shape_2d(const TensorDesc& a, const TensorDesc& b) {
  return is_rank(a, 2) && is_rank(b, 2) &&
         a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1];
}

// GEMM contract: A[M,K], B[K,N], C[M,N]
inline bool gemm_shape_ok_2d(const TensorDesc& A,
                            const TensorDesc& B,
                            const TensorDesc& C) {
  if (!is_rank(A, 2) || !is_rank(B, 2) || !is_rank(C, 2)) return false;

  const int64_t M  = A.shape[0];
  const int64_t K  = A.shape[1];
  const int64_t K2 = B.shape[0];
  const int64_t N  = B.shape[1];

  if (M <= 0 || N <= 0 || K <= 0) return false;
  if (K2 != K) return false;
  if (C.shape[0] != M || C.shape[1] != N) return false;

  return true;
}

} // namespace aicf::cuda::shim
