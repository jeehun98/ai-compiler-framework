#pragma once

#include <cstdint>
#include <cstddef>

namespace aicf::cuda {

constexpr int kMaxRank = 4;

enum class DType : uint8_t {
  kF32  = 0,
  kF16  = 1,
  kBF16 = 2,

  // backward-compatible aliases (keep)
  F32  = kF32,
  F16  = kF16,
  BF16 = kBF16,
};

struct TensorDesc {
  void* data = nullptr;
  DType dtype = DType::kF32;

  // named union for MSVC compatibility
  union RankND {
    int32_t rank;
    int32_t ndim; // alias field (ok)
    RankND() : rank(0) {}
  } r;

  int64_t shape[kMaxRank]  = {0, 0, 0, 0};
  int64_t stride[kMaxRank] = {0, 0, 0, 0};

  bool contiguous = false;
  int32_t alignment = 0;
  int32_t device = 0;

  // Canonical accessor: rank only.
  // NOTE: keep this name distinct from "ndim" to avoid field/method confusion.
  int32_t rank() const { return r.rank; }

  // v0.2: REMOVE ndim() to avoid accidental "t.ndim" binding-to-method issues.
  // If you really need it, use t.r.ndim or t.rank().
  //
  // int32_t ndim() const { return r.ndim; }  // <-- DO NOT KEEP
};

} // namespace aicf::cuda
