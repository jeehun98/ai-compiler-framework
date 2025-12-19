// aicf/backends/cuda/registry/attr_pack.hpp (신규)
#pragma once
#include <cstdint>
#include <string_view>

namespace aicf::cuda {

enum class AttrTag : int32_t { kI64 = 0, kF32 = 1, kBool = 2 };

struct AttrValue {
  AttrTag tag;
  union {
    int64_t i64;
    float   f32;
    int32_t b32;
  };
};

struct AttrKV {
  const char* key;  // null-terminated
  AttrValue   val;
};

struct AttrPack {
  const AttrKV* items = nullptr;
  int32_t size = 0;

  bool get_i64(std::string_view k, int64_t* out) const;
  bool get_f32(std::string_view k, float* out) const;
  bool get_bool(std::string_view k, bool* out) const;
};

} // namespace aicf::cuda
