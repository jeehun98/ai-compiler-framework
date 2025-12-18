#include <cuda_runtime.h>
#include "aicf/backends/cuda/registry/register_all.hpp"

extern "C" __global__ void aicf_cuda_anchor_kernel() {}
void aicf_cuda_context_init() {
  // 기존 init 로직들...

  // 커널 레지스트리 초기화 (한 번만)
  aicf_cuda_register_all_kernels();

  // 이후 로직...
}
extern "C" void aicf_cuda_anchor() {
    // 함수 "주소"를 명시적으로 참조
    auto fp = &aicf_cuda_anchor_kernel;
    (void)fp;
}
