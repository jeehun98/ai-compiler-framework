#pragma once
#include <cuda_runtime.h>

#include "aicf/runtime/stream.hpp"

namespace aicf::cuda::shim {

// aicf::Stream -> cudaStream_t
inline cudaStream_t to_cuda_stream(aicf::Stream s) {
  return (s.handle == nullptr) ? (cudaStream_t)0 : reinterpret_cast<cudaStream_t>(s.handle);
}

// cudaStream_t -> aicf::Stream
inline aicf::Stream from_cuda_stream(cudaStream_t s) {
  aicf::Stream out{};
  out.handle = reinterpret_cast<void*>(s);
  return out;
}

} // namespace aicf::cuda::shim
