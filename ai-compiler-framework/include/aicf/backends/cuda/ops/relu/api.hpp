#pragma once
#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

namespace aicf::cuda {

aicf::Status relu_f32(const float* in, float* out, int N, aicf::Stream stream);

// NEW (v0.2): f16
aicf::Status relu_f16(const void* in, void* out, int N, aicf::Stream stream);

} // namespace aicf::cuda
