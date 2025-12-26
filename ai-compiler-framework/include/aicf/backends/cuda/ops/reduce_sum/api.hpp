#pragma once
#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

namespace aicf::cuda {

// reduce sum over axis=0 for a flattened (M,N) view:
//   In:  dY [M,N]
//   Out: dB [N]
// Meaning: dB[j] = sum_i dY[i,j]
aicf::Status reduce_sum_lastdim_f32(const float* dY,
                                   float* dB,
                                   int M, int N,
                                   aicf::Stream stream);

} // namespace aicf::cuda
