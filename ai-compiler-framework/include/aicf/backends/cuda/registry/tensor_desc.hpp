#pragma once
#include <cstdint>
#include <cstddef>

namespace aicf::cuda {

constexpr int kMaxRank = 4;

enum class DType : uint8_t {
  kF32  = 0,
  kF16  = 1,
  kBF16 = 2,

  // backward-compatible aliases
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
    int32_t ndim;
    RankND() : rank(0) {}
  } r;

  int64_t shape[kMaxRank]  = {0, 0, 0, 0};
  int64_t stride[kMaxRank] = {0, 0, 0, 0};

  bool contiguous = false;
  int32_t alignment = 0;
  int32_t device = 0;

  // convenience accessors (선택)
  int32_t rank() const { return r.rank; }
  int32_t ndim() const { return r.ndim; }
};

} // namespace aicf::cuda
