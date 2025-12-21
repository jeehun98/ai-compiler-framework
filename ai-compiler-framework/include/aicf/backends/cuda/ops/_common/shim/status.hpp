#pragma once
#include <cuda_runtime.h>

#include "aicf/core/status.hpp"

namespace aicf::cuda::shim {

// 커널 launch 이후 표준 에러 체크 규칙
inline aicf::Status cuda_last_error_to_status() {
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? aicf::Status::Ok : aicf::Status::Error;
}

} // namespace aicf::cuda::shim
