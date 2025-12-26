#pragma once
#include <cstdint>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

namespace aicf::cuda {

// param = param - lr * grad
aicf::Status sgd_step_f32(float* param,
                          const float* grad,
                          int64_t numel,
                          float lr,
                          aicf::Stream stream);

} // namespace aicf::cuda
