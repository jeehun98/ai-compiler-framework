#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

namespace aicf {
namespace cuda {
namespace ops {

template <typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename T>
__global__ void softmax_fwd_kernel(const T* x, T* y, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // 1. Max Reduction
    float max_val = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x) {
        max_val = max(max_val, (float)x[row * cols + i]);
    }
    max_val = warpReduceMax(max_val); // 단순화를 위해 blockDim을 32로 가정하거나 shared mem 사용

    // 2. Exp and Sum Reduction
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = expf((float)x[row * cols + i] - max_val);
        y[row * cols + i] = (T)val;
        sum += val;
    }
    sum = warpReduceSum(sum);

    // 3. Final Division
    for (int i = tid; i < cols; i += blockDim.x) {
        y[row * cols + i] = (T)((float)y[row * cols + i] / sum);
    }
}

} // namespace ops
} // namespace cuda
} // namespace aicf