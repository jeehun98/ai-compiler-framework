#pragma once

#include <cstdint>
#include <string_view>

#include "aicf/backends/cuda/registry/attr_pack.hpp"

namespace aicf::cuda {

// Contract:
// - attr pointer is either nullptr or points to AttrPack
// - if nullptr => treated as empty
inline const AttrPack& as_attr_pack(const void* attr) {
  static const AttrPack kEmpty{};
  return attr ? *reinterpret_cast<const AttrPack*>(attr) : kEmpty;
}

// "Optional with default" helpers.
// Type mismatch -> treated as missing (returns default).
inline int64_t attr_i64(const void* attr, std::string_view key, int64_t def) {
  const AttrPack& p = as_attr_pack(attr);
  int64_t v = 0;
  return p.get_i64(key, &v) ? v : def;
}

inline float attr_f32(const void* attr, std::string_view key, float def) {
  const AttrPack& p = as_attr_pack(attr);
  float v = 0.0f;
  return p.get_f32(key, &v) ? v : def;
}

inline bool attr_bool(const void* attr, std::string_view key, bool def) {
  const AttrPack& p = as_attr_pack(attr);
  bool v = false;
  return p.get_bool(key, &v) ? v : def;
}

// "Required" helpers.
// Return false when missing OR wrong type.
inline bool attr_require_i64(const void* attr, std::string_view key, int64_t* out) {
  const AttrPack& p = as_attr_pack(attr);
  return p.get_i64(key, out);
}
inline bool attr_require_f32(const void* attr, std::string_view key, float* out) {
  const AttrPack& p = as_attr_pack(attr);
  return p.get_f32(key, out);
}
inline bool attr_require_bool(const void* attr, std::string_view key, bool* out) {
  const AttrPack& p = as_attr_pack(attr);
  return p.get_bool(key, out);
}

} // namespace aicf::cuda
