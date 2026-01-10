#pragma once

#include <cuda_runtime.h>

#include <aicf/core/status.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>

namespace aicf::cuda {

// StepInc: int32 scalar tensor (+1). in-place allowed.
// inputs : [step]  (numel==1, int32, cuda)
// outputs: [step]  (same)
aicf::Status step_inc_v0(
    const aicf::cuda::TensorDesc* inputs,  int in_n,
    const aicf::cuda::TensorDesc* outputs, int out_n,
    const void* attr_pack,
    cudaStream_t stream);

} // namespace aicf::cuda
