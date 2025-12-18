#pragma once

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

namespace aicf::cuda {

// out[N] = a[N] + b[N]
// - contiguous 1D
// - stream.handle == nullptr => default stream(0)
aicf::Status add_f32(const float* a,
                     const float* b,
                     float* out,
                     int N,
                     aicf::Stream stream);

} // namespace aicf::cuda
