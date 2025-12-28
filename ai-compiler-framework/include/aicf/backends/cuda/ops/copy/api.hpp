#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#include "aicf/core/status.hpp"
#include "aicf/backends/cuda/registry/tensor_desc.hpp"

namespace aicf::cuda {

// v0.2 KernelVariant contract-compatible:
// outputs is non-const, workspace args exist (may be nullptr/0 in v0.1 policy)
aicf::Status copy_v0(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream);

} // namespace aicf::cuda
