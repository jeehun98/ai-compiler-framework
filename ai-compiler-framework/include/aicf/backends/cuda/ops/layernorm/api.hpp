#pragma once

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

namespace aicf::cuda {

// LayerNorm forward (2D, last-dim normalize)
// x: [M, N]
// gamma/beta: [N] (optional; can be nullptr)
// y: [M, N]
// mean/rstd: [M]  (rstd = 1/sqrt(var + eps))  -> backward 재사용용
//
// - stream.handle == nullptr => default stream(0)
// - All pointers must be device pointers
Status layernorm_fwd_f32(const float* x,
                         const float* gamma,  // nullable
                         const float* beta,   // nullable
                         float* y,
                         float* mean,
                         float* rstd,
                         int M, int N,
                         float eps,
                         Stream stream);

// Half input/output, mean/rstd are float
Status layernorm_fwd_f16(const void* x,       // __half*
                         const void* gamma,   // __half* nullable
                         const void* beta,    // __half* nullable
                         void* y,             // __half*
                         float* mean,
                         float* rstd,
                         int M, int N,
                         float eps,
                         Stream stream);

// (선택) backward는 PR 다음 단계에서 닫아도 됨.
// 시그니처만 먼저 박아두면 python/registry 연결이 쉬움.
Status layernorm_bwd_f32(float* dx,
                         float* dgamma,   // nullable
                         float* dbeta,    // nullable
                         const float* dy,
                         const float* x,
                         const float* mean,
                         const float* rstd,
                         const float* gamma, // nullable
                         int M, int N,
                         Stream stream);

Status layernorm_bwd_f16(void* dx,            // __half*
                         void* dgamma,        // __half* nullable
                         void* dbeta,         // __half* nullable
                         const void* dy,      // __half*
                         const void* x,       // __half*
                         const float* mean,
                         const float* rstd,
                         const void* gamma,   // __half* nullable
                         int M, int N,
                         Stream stream);

} // namespace aicf::cuda
