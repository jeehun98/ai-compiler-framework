#pragma once

#include <cstdint>
#include <cstddef>

namespace aicf::cuda {

constexpr int kMaxRank = 4;

// NOTE:
// - kUnknown is intentionally included for "not-yet-resolved" dtype at IR/registry stages.
// - Keep backward-compatible aliases.
enum class DType : uint8_t {
  kUnknown = 0,

  kF32  = 1,
  kF16  = 2,
  kBF16 = 3,
  kI32  = 4,   // ✅ int32

  // backward-compatible aliases (keep)
  Unknown = kUnknown,
  F32  = kF32,
  F16  = kF16,
  BF16 = kBF16,
  I32  = kI32, // ✅ alias
};

// -------------------------
// DType helpers
// -------------------------
static inline constexpr bool dtype_is_known(DType t) {
  return t != DType::kUnknown;
}

static inline constexpr bool dtype_is_float(DType t) {
  return (t == DType::kF32) || (t == DType::kF16) || (t == DType::kBF16);
}

static inline constexpr bool dtype_is_int(DType t) {
  return (t == DType::kI32);
}

static inline constexpr bool dtype_is_supported(DType t) {
  // extend here when you add int8, etc.
  return dtype_is_float(t) || dtype_is_int(t) || (t == DType::kUnknown);
}

static inline constexpr int32_t dtype_bits(DType t) {
  switch (t) {
    case DType::kF32:  return 32;
    case DType::kF16:  return 16;
    case DType::kBF16: return 16;
    case DType::kI32:  return 32;  // ✅
    default:           return 0;
  }
}

static inline constexpr int32_t dtype_size_bytes(DType t) {
  return dtype_bits(t) / 8;
}

static inline constexpr const char* dtype_name(DType t) {
  switch (t) {
    case DType::kUnknown: return "unknown";
    case DType::kF32:     return "f32";
    case DType::kF16:     return "f16";
    case DType::kBF16:    return "bf16";
    case DType::kI32:     return "i32";     // ✅
    default:              return "invalid";
  }
}

// Common accumulation rule (useful for GEMM/Reductions)
static inline constexpr DType dtype_accum_for(DType t) {
  switch (t) {
    case DType::kF16:  return DType::kF32;
    case DType::kBF16: return DType::kF32;
    case DType::kF32:  return DType::kF32;
    case DType::kI32:  return DType::kI32;  // ✅ (정수는 정수 누산이 자연스러움)
    default:           return DType::kF32;
  }
}

// -------------------------
// TensorDesc
// -------------------------
struct TensorDesc {
  void* data = nullptr;
  DType dtype = DType::kUnknown;

  // named union for MSVC compatibility
  union RankND {
    int32_t rank;
    int32_t ndim; // alias field (ok)
    RankND() : rank(0) {}
  } r;

  // shape/stride are in elements
  int64_t shape[kMaxRank]  = {0, 0, 0, 0};
  int64_t stride[kMaxRank] = {0, 0, 0, 0};

  bool contiguous = false;
  int32_t alignment = 0; // bytes; 0 = unknown
  int32_t device = 0;

  // Canonical accessor: rank only.
  int32_t rank() const { return r.rank; }

  // -------------------------
  // Convenience helpers
  // -------------------------
  bool is_valid_rank() const {
    return rank() >= 0 && rank() <= kMaxRank;
  }

  bool has_data() const { return data != nullptr; }

  bool has_valid_strides_and_shapes() const {
    // rank==0(스칼라)은 shape/stride 검증을 요구하지 않는다.
    if (!is_valid_rank()) return false;
    if (rank() == 0) return true;
    for (int32_t i = 0; i < rank(); ++i) {
      if (shape[i] <= 0) return false;
      if (stride[i] <= 0) return false;
    }
    return true;
  }

  int64_t numel() const {
    // ✅ 기존 구현은 rank<=0이면 0이었을 가능성이 큼.
    // torch scalar(rank=0)를 안전하게 지원하려면 numel=1이 맞다.
    if (!is_valid_rank()) return 0;
    if (rank() == 0) return 1;
    int64_t n = 1;
    for (int32_t i = 0; i < rank(); ++i) n *= shape[i];
    return n;
  }

  int64_t nbytes() const {
    return numel() * (int64_t)dtype_size_bytes(dtype);
  }

  bool aligned_at_least(int32_t bytes) const {
    if (bytes <= 0) return true;
    if (alignment > 0) return alignment >= bytes;
    return false; // unknown alignment => don't claim
  }

  // 2D helpers (row-major contiguous)
  bool is_2d() const { return rank() == 2; }

  bool is_contig_2d_rowmajor() const {
    if (!is_2d()) return false;
    // element-strides
    return (stride[1] == 1) && (stride[0] == shape[1]);
  }
};

} // namespace aicf::cuda
