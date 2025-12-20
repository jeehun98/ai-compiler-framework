#pragma once
#include <cstdint>
#include <cstddef>   // size_t

#include "aicf/backends/cuda/registry/tensor_desc.hpp"

namespace aicf::cuda::shim {

// -------- rank (canonical) --------
inline int32_t rank(const TensorDesc& t) { return t.r.rank; }
inline bool is_rank(const TensorDesc& t, int32_t n) { return rank(t) == n; }

// -------- basics --------
inline bool is_contig(const TensorDesc& t) { return t.contiguous; }

inline bool is_f32(const TensorDesc& t) { return t.dtype == DType::F32; }
inline bool is_f16(const TensorDesc& t) { return t.dtype == DType::F16; }
inline bool is_bf16(const TensorDesc& t) { return t.dtype == DType::BF16; }

// -------- presets --------
inline bool is_f32_contig_1d(const TensorDesc& t) { return is_f32(t) && is_contig(t) && is_rank(t, 1); }
inline bool is_f32_contig_2d(const TensorDesc& t) { return is_f32(t) && is_contig(t) && is_rank(t, 2); }
inline bool is_f16_contig_1d(const TensorDesc& t) { return is_f16(t) && is_contig(t) && is_rank(t, 1); }
inline bool is_f16_contig_2d(const TensorDesc& t) { return is_f16(t) && is_contig(t) && is_rank(t, 2); }

// -------- shape helpers --------
inline bool same_shape_1d(const TensorDesc& a, const TensorDesc& b) {
  return is_rank(a, 1) && is_rank(b, 1) && a.shape[0] == b.shape[0];
}

inline bool same_shape_2d(const TensorDesc& a, const TensorDesc& b) {
  return is_rank(a, 2) && is_rank(b, 2) &&
         a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1];
}

// GEMM contract: A[M,K], B[K,N], C[M,N]
inline bool gemm_shape_ok_2d(const TensorDesc& A, const TensorDesc& B, const TensorDesc& C) {
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

// ============================================================
// v0.2 extensions (NEW)
// ============================================================

// -------- alignment helpers --------
inline bool is_aligned(const void* p, size_t bytes) {
  // bytes must be power-of-two for this to be meaningful
  const uintptr_t x = reinterpret_cast<uintptr_t>(p);
  return (x & (static_cast<uintptr_t>(bytes) - 1u)) == 0u;
}

inline bool is_aligned_data(const TensorDesc& t, size_t bytes) {
  return is_aligned(t.data, bytes);
}

// -------- numel helpers (rank <= kMaxRank=4) --------
inline int64_t numel(const TensorDesc& t) {
  const int32_t r = rank(t);
  if (r <= 0) return 0;
  int64_t n = 1;
  for (int i = 0; i < r; ++i) {
    const int64_t d = t.shape[i];
    if (d <= 0) return 0;
    n *= d;
  }
  return n;
}

// -------- vectorization helpers --------
// Common for half2/float2/float4 etc.
inline bool is_even_len_1d(const TensorDesc& t) {
  return is_rank(t, 1) && (t.shape[0] > 0) && ((t.shape[0] & 1) == 0);
}

} // namespace aicf::cuda::shim
