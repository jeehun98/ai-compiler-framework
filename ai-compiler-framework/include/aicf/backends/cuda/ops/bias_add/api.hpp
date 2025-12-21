#pragma once

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

namespace aicf::cuda {

// Y[M,N] + bias[N] -> Out[M,N]
aicf::Status bias_add_f32(const float* Y,
                          const float* bias,
                          float* Out,
                          int M, int N,
                          aicf::Stream stream);

} // namespace aicf::cuda
