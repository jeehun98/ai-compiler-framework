#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#include "aicf/backends/cuda/registry/status.hpp"
#include "aicf/backends/cuda/registry/op_kind.hpp"
#include "aicf/backends/cuda/registry/tensor_desc.hpp"

namespace aicf::cuda {

// Minimal C ABI-style wrapper used by bindings
Status dispatch_v0(
    OpKind kind,
    const TensorDesc* inputs, int32_t num_inputs,
    TensorDesc* outputs, int32_t num_outputs,
    const void* attrs,
    cudaStream_t stream);

} // namespace aicf::cuda
