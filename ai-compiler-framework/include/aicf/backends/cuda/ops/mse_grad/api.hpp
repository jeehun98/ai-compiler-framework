#pragma once
#include <cstdint>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

namespace aicf::cuda {

// dPred = scale * (pred - target)
// scale is typically 2.0f / numel for MSE(mean) gradient.
aicf::Status mse_grad_f32(const float* pred,
                          const float* target,
                          float* dPred,
                          int64_t numel,
                          float scale,
                          aicf::Stream stream);

} // namespace aicf::cuda
