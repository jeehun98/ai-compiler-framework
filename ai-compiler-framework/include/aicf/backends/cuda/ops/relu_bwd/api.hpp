#pragma once
#include <cstdint>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

namespace aicf::cuda {

// dY = (Y > 0) ? dOut : 0
aicf::Status relu_bwd_f32(const float* Y,
                          const float* dOut,
                          float* dY,
                          int64_t numel,
                          aicf::Stream stream);

} // namespace aicf::cuda
