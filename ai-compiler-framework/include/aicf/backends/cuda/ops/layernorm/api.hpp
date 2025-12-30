#pragma once

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

namespace aicf::cuda {

// Forward
// x: [M,N], gamma/beta: [N] (optional), y:[M,N], mean:[M], rstd:[M]
Status layernorm_fwd_f32(const float* x,
                         const float* gamma,
                         const float* beta,
                         float* y,
                         float* mean,
                         float* rstd,
                         int M, int N,
                         float eps,
                         Stream stream);

Status layernorm_fwd_f16(const void* x,
                         const void* gamma,
                         const void* beta,
                         void* y,
                         float* mean,
                         float* rstd,
                         int M, int N,
                         float eps,
                         Stream stream);

// Backward
// affine=True:
//   inputs : x[M,N], dy[M,N], gamma[N], mean[M], rstd[M]
//   outputs: dx[M,N], dgamma[N](f32), dbeta[N](f32)
// affine=False:
//   inputs : x[M,N], dy[M,N], mean[M], rstd[M]
//   outputs: dx[M,N]
Status layernorm_bwd_f32(float* dx, float* dgamma, float* dbeta,
                         const float* x, const float* dy,
                         const float* gamma,
                         const float* mean, const float* rstd,
                         int M, int N,
                         Stream stream);

Status layernorm_bwd_f16(void* dx, float* dgamma, float* dbeta,
                         const void* x, const void* dy,
                         const void* gamma,
                         const float* mean, const float* rstd,
                         int M, int N,
                         Stream stream);

} // namespace aicf::cuda
