#pragma once

namespace aicf::cuda {

enum class OpKind : int {
  EltwiseAdd  = 0,
  EltwiseRelu = 1,
  Gemm        = 2,
  BiasAdd     = 3,
  ReduceSum   = 4,
  _Count      = 5
};

}  // namespace aicf::cuda
