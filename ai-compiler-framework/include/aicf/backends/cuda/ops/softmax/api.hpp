#pragma once
#include <aicf/backends/cuda/registry/status.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <cuda_runtime.h>

namespace aicf {
namespace cuda {
namespace ops {

Status softmax_fwd(
    const TensorDesc& x_desc, const void* x_ptr,
    const TensorDesc& y_desc, void* y_ptr,
    int64_t axis,
    cudaStream_t stream);

} // namespace ops
} // namespace cuda
} // namespace aicf