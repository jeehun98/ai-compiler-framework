#include <cuda_runtime.h>

extern "C" __global__ void aicf_cuda_anchor_kernel() {}

extern "C" void aicf_cuda_anchor() {
    // 함수 "주소"를 명시적으로 참조
    auto fp = &aicf_cuda_anchor_kernel;
    (void)fp;
}
