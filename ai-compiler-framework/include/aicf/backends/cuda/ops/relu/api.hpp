#pragma once

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

namespace aicf::cuda {

// out[N] = max(in[N], 0)
// - contiguous 1D
// - stream.handle == nullptr => default stream(0)
aicf::Status relu_f32(const float* in,
                      float* out,
                      int N,
                      aicf::Stream stream);

} // namespace aicf::cuda
