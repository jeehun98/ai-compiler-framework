// aicf/backends/cuda/registry/attr_pack.hpp
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

  // Returns true only when key exists AND tag matches.
  inline bool get_i64(std::string_view k, int64_t* out) const {
    if (!out) return false;
    if (!items || size <= 0) return false;
    for (int32_t i = 0; i < size; ++i) {
      const AttrKV& kv = items[i];
      if (!kv.key) continue;
      if (k == std::string_view(kv.key)) {
        if (kv.val.tag != AttrTag::kI64) return false;
        *out = kv.val.i64;
        return true;
      }
    }
    return false;
  }

  inline bool get_f32(std::string_view k, float* out) const {
    if (!out) return false;
    if (!items || size <= 0) return false;
    for (int32_t i = 0; i < size; ++i) {
      const AttrKV& kv = items[i];
      if (!kv.key) continue;
      if (k == std::string_view(kv.key)) {
        if (kv.val.tag != AttrTag::kF32) return false;
        *out = kv.val.f32;
        return true;
      }
    }
    return false;
  }

  inline bool get_bool(std::string_view k, bool* out) const {
    if (!out) return false;
    if (!items || size <= 0) return false;
    for (int32_t i = 0; i < size; ++i) {
      const AttrKV& kv = items[i];
      if (!kv.key) continue;
      if (k == std::string_view(kv.key)) {
        if (kv.val.tag != AttrTag::kBool) return false;
        *out = (kv.val.b32 != 0);
        return true;
      }
    }
    return false;
  }
};

} // namespace aicf::cuda
