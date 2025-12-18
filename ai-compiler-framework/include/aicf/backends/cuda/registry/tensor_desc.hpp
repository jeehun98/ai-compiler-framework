#pragma once
#include <cstdint>

namespace aicf::cuda {

// 초기에는 F32만 쓰더라도, dtype은 확장 대비로 남겨둠.
enum class DType : uint8_t {
  F32 = 0,
  F16 = 1,
  BF16 = 2
};

// 초기 최소 구현: rank<=4, contiguous 중심
struct TensorDesc {
  void* data = nullptr;
  DType dtype = DType::F32;

  int ndim = 0;
  int64_t shape[4] = {0, 0, 0, 0};
  int64_t stride[4] = {0, 0, 0, 0};

  bool contiguous = true;
  int alignment = 0;  // bytes: 4,8,16...
  int device = 0;
};

}  // namespace aicf::cuda
