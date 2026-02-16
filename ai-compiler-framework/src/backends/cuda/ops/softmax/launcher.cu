#include <aicf/backends/cuda/ops/softmax/api.hpp>
#include "kernels.cuh"

namespace aicf {
namespace cuda {
namespace ops {

Status softmax_fwd(const TensorDesc& x_desc, const void* x_ptr,
                  const TensorDesc& y_desc, void* y_ptr,
                  int64_t axis, cudaStream_t stream) {
    // 단순화를 위해 마지막 차원(cols) 기준 softmax 가정
    int64_t rows = 1;
    for(int i=0; i<axis; ++i) rows *= x_desc.shape[i];
    int64_t cols = x_desc.shape[axis];

    int threads = 128; // 적절한 thread 수 선택
    softmax_fwd_kernel<float><<<rows, threads, 0, stream>>>(
        (const float*)x_ptr, (float*)y_ptr, rows, cols);

    return Status::Ok();
}

} // namespace ops
} // namespace cuda
} // namespace aicf