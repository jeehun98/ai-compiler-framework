#pragma once

#include <cuda_runtime.h>

#include "aicf/core/status.hpp"
#include "aicf/backends/cuda/registry/op_kind.hpp"
#include "aicf/backends/cuda/registry/registry.hpp"   // OpCall + Dispatch()

namespace aicf::cuda {

// Plan A / v0.1 entry wrapper:
// - workspace 없음
// - attr는 nullptr 또는 AttrPack* 로 전달 (registry.cpp에서 empty AttrPack로 치환)
inline aicf::Status dispatch_v0(
    OpKind kind,
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const AttrPack* attrs,
    cudaStream_t stream) {

  OpCall call{};
  call.kind = kind;
  call.inputs = inputs;
  call.num_inputs = num_inputs;
  call.outputs = outputs;
  call.num_outputs = num_outputs;
  call.attrs = attrs;
  call.stream = stream;

  return Dispatch(call);
}

// attr를 void*로 갖고 있는 기존 호출부가 있으면 이 overload도 유용함
inline aicf::Status dispatch_v0(
    OpKind kind,
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,         // nullptr or AttrPack*
    cudaStream_t stream) {

  return dispatch_v0(kind,
                     inputs, num_inputs,
                     outputs, num_outputs,
                     reinterpret_cast<const AttrPack*>(attr),
                     stream);
}

} // namespace aicf::cuda
