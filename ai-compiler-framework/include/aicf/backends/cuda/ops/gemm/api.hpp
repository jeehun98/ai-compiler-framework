#pragma once

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

namespace aicf::cuda {

// C[MxN] = A[MxK] * B[KxN]  (row-major)
// - stream.handle == nullptr 이면 default stream(0) 사용
Status gemm_f32(const float* A,
                const float* B,
                float* C,
                int M, int N, int K,
                Stream stream);

} // namespace aicf::cuda
