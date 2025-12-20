#pragma once
#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

namespace aicf::cuda {

aicf::Status add_f32(const float* a, const float* b, float* out, int N, aicf::Stream stream);

// NEW (v0.2): f16
aicf::Status add_f16(const void* a, const void* b, void* out, int N, aicf::Stream stream);
// 또는 __half를 노출해도 되지만, 헤더에 cuda_fp16 포함 싫으면 void*가 깔끔함.

} // namespace aicf::cuda
