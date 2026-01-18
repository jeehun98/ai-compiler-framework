#pragma once
#include <cstddef>
#include <cstdint>

namespace aicf::cuda {

// AttrBlob = "opaque attribute payload" (ABI stable)
//
// Rules:
// - schema_id == 0 means "unspecified" (ops may default)
// - bytes may be 0 and data may be nullptr
// - payload must be POD layout if interpreted by kernels
struct AttrBlob {
  uint32_t schema_id = 0;
  uint32_t bytes = 0;
  const void* data = nullptr;
};

} // namespace aicf::cuda
