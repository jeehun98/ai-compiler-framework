#pragma once

namespace aicf::cuda {

enum class OpKind : int {
  EltwiseAdd  = 0,
  EltwiseRelu = 1,
  Gemm        = 2,
  BiasAdd     = 3,
  ReduceSum   = 4,
  MseGrad     = 5,
  ReluBwd     = 6,
  _Count      = 7
};

}  // namespace aicf::cuda
