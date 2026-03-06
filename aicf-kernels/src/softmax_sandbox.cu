#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

// sandbox 환경을 위한 매크로 및 타이머
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void tic() { cudaEventRecord(start); }
    float toc() {
        float ms;
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ============================================================================
// Softmax Kernels (From aicf::cuda)
// ============================================================================
namespace softmax_impl {

static __forceinline__ __device__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float oth = __shfl_down_sync(0xffffffffu, v, offset);
        v = (v > oth) ? v : oth;
    }
    return v;
}

static __forceinline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

static __forceinline__ __device__ float block_reduce_max(float v) {
    __shared__ float smem[32]; 
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    v = warp_reduce_max(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();

    float out = -INFINITY;
    if (warp == 0) {
        int warps = (blockDim.x + 31) >> 5;
        out = (lane < warps) ? smem[lane] : -INFINITY;
        out = warp_reduce_max(out);
    }
    if (threadIdx.x == 0) smem[0] = out;
    __syncthreads();
    return smem[0];
}

static __forceinline__ __device__ float block_reduce_sum(float v) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();

    float out = 0.0f;
    if (warp == 0) {
        int warps = (blockDim.x + 31) >> 5;
        out = (lane < warps) ? smem[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }
    if (threadIdx.x == 0) smem[0] = out;
    __syncthreads();
    return smem[0];
}

__global__ void softmax_lastdim_f32_kernel(const float* __restrict__ x,
                                           float* __restrict__ y,
                                           int64_t rows, int64_t cols) {
    const int64_t row = (int64_t)blockIdx.x;
    if (row >= rows) return;
    const int64_t base = row * cols;

    float tmax = -INFINITY;
    for (int64_t c = (int64_t)threadIdx.x; c < cols; c += (int64_t)blockDim.x) {
        tmax = fmaxf(tmax, x[base + c]);
    }
    const float rmax = block_reduce_max(tmax);

    float tsum = 0.0f;
    for (int64_t c = (int64_t)threadIdx.x; c < cols; c += (int64_t)blockDim.x) {
        tsum += __expf(x[base + c] - rmax);
    }
    const float rsum = block_reduce_sum(tsum);

    const float inv = 1.0f / rsum;
    for (int64_t c = (int64_t)threadIdx.x; c < cols; c += (int64_t)blockDim.x) {
        y[base + c] = __expf(x[base + c] - rmax) * inv;
    }
}

__global__ void softmax_lastdim_f16_kernel(const __half* __restrict__ x,
                                           __half* __restrict__ y,
                                           int64_t rows, int64_t cols) {
    const int64_t row = (int64_t)blockIdx.x;
    if (row >= rows) return;
    const int64_t base = row * cols;

    float tmax = -INFINITY;
    for (int64_t c = (int64_t)threadIdx.x; c < cols; c += (int64_t)blockDim.x) {
        float v = __half2float(x[base + c]);
        tmax = fmaxf(tmax, v);
    }
    const float rmax = block_reduce_max(tmax);

    float tsum = 0.0f;
    for (int64_t c = (int64_t)threadIdx.x; c < cols; c += (int64_t)blockDim.x) {
        float v = __half2float(x[base + c]);
        tsum += __expf(v - rmax);
    }
    const float rsum = block_reduce_sum(tsum);

    const float inv = 1.0f / rsum;
    for (int64_t c = (int64_t)threadIdx.x; c < cols; c += (int64_t)blockDim.x) {
        float v = __half2float(x[base + c]);
        y[base + c] = __float2half_rn(__expf(v - rmax) * inv);
    }
}

} // namespace softmax_impl

// -------------------------
// CLI Helpers
// -------------------------
static inline int arg_int(int argc, char** argv, const char* key, int def) {
    for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == key) return std::atoi(argv[i + 1]);
    return def;
}
static inline std::string arg_str(int argc, char** argv, const char* key, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == key) return argv[i + 1];
    return def;
}

// -------------------------
// Validation
// -------------------------
void validate_softmax(int rows, int cols, const float* h_X, const void* d_Y, std::string dtype) {
    float max_err = 0.0f;
    std::vector<float> h_Y(rows * cols);
    
    if (dtype == "f32") {
        CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        std::vector<__half> temp(rows * cols);
        CUDA_CHECK(cudaMemcpy(temp.data(), d_Y, rows * cols * sizeof(__half), cudaMemcpyDeviceToHost));
        for(int i=0; i<rows*cols; ++i) h_Y[i] = __half2float(temp[i]);
    }

    for (int r = 0; r < rows; ++r) {
        float rmax = -INFINITY;
        for (int c = 0; c < cols; ++c) rmax = std::max(rmax, h_X[r * cols + c]);
        float rsum = 0.0f;
        for (int c = 0; c < cols; ++c) rsum += std::exp(h_X[r * cols + c] - rmax);
        
        for (int c = 0; c < cols; ++c) {
            float ref = std::exp(h_X[r * cols + c] - rmax) / rsum;
            max_err = std::max(max_err, std::abs(ref - h_Y[r * cols + c]));
        }
    }
    printf("max_abs_error: %f\n", max_err);
}

int main(int argc, char** argv) {
    const int rows = arg_int(argc, argv, "--rows", 1024);
    const int cols = arg_int(argc, argv, "--cols", 1024);
    const int iters = arg_int(argc, argv, "--iters", 100);
    const int warmup = arg_int(argc, argv, "--warmup", 10);
    const std::string dtype = arg_str(argc, argv, "--dtype", "f32");

    printf("Softmax: rows=%d cols=%d dtype=%s iters=%d\n", rows, cols, dtype.c_str(), iters);

    std::vector<float> h_X(rows * cols);
    for (int i = 0; i < rows * cols; ++i) h_X[i] = (float)(rand() % 10) - 5.0f;

    GpuTimer timer;
    if (dtype == "f32") {
        float *d_X, *d_Y;
        CUDA_CHECK(cudaMalloc(&d_X, rows * cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Y, rows * cols * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));

        for (int i = 0; i < warmup; ++i) softmax_impl::softmax_lastdim_f32_kernel<<<rows, 256>>>(d_X, d_Y, rows, cols);
        timer.tic();
        for (int i = 0; i < iters; ++i) softmax_impl::softmax_lastdim_f32_kernel<<<rows, 256>>>(d_X, d_Y, rows, cols);
        float ms = timer.toc() / iters;

        validate_softmax(rows, cols, h_X.data(), d_Y, "f32");
        printf("avg_kernel_ms: %f\n", ms);
        cudaFree(d_X); cudaFree(d_Y);
    } else {
        __half *d_X, *d_Y;
        std::vector<__half> h_Xh(rows * cols);
        for(int i=0; i<rows*cols; ++i) h_Xh[i] = __float2half(h_X[i]);

        CUDA_CHECK(cudaMalloc(&d_X, rows * cols * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_Y, rows * cols * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_X, h_Xh.data(), rows * cols * sizeof(__half), cudaMemcpyHostToDevice));

        for (int i = 0; i < warmup; ++i) softmax_impl::softmax_lastdim_f16_kernel<<<rows, 256>>>(d_X, d_Y, rows, cols);
        timer.tic();
        for (int i = 0; i < iters; ++i) softmax_impl::softmax_lastdim_f16_kernel<<<rows, 256>>>(d_X, d_Y, rows, cols);
        float ms = timer.toc() / iters;

        validate_softmax(rows, cols, h_X.data(), d_Y, "f16");
        printf("avg_kernel_ms: %f\n", ms);
        cudaFree(d_X); cudaFree(d_Y);
    }

    return 0;
}