#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

// sandbox 환경용 유틸리티
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
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
// CrossEntropyLoss Kernels (From launcher.cu)
// ============================================================================
namespace xent_impl {

__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__device__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    return v;
}

__device__ float block_sum(float v) {
    __shared__ float sh[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    float out = (wid == 0) ? ((lane < ((blockDim.x + 31) >> 5)) ? sh[lane] : 0.0f) : 0.0f;
    if (wid == 0) {
        out = warp_sum(out);
        if (lane == 0) sh[0] = out;
    }
    __syncthreads();
    return sh[0];
}

__device__ float block_max(float v) {
    __shared__ float sh[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    float out = (wid == 0) ? ((lane < ((blockDim.x + 31) >> 5)) ? sh[lane] : -1e38f) : -1e38f;
    if (wid == 0) {
        out = warp_max(out);
        if (lane == 0) sh[0] = out;
    }
    __syncthreads();
    return sh[0];
}

__global__ void xent_fwd_sum_f32(const float* logits, const int* targets, int N, int C, int ignore_index, float* out_loss, int* out_valid) {
    int n = blockIdx.x;
    if (n >= N || targets[n] == ignore_index) return;
    int t = targets[n];
    const float* row = logits + (size_t)n * C;

    float m = -1e38f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) m = fmaxf(m, row[c]);
    m = block_max(m);

    float s = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) s += expf(row[c] - m);
    s = block_sum(s);

    if (threadIdx.x == 0 && s > 0.0f) {
        atomicAdd(out_loss, logf(s) + m - row[t]);
        atomicAdd(out_valid, 1);
    }
}

__global__ void xent_finalize_loss_f32(const float* loss_sum, const int* valid, int reduction, float* out_loss) {
    if (threadIdx.x == 0) {
        if (reduction == 0) out_loss[0] = (valid[0] > 0) ? (loss_sum[0] / (float)valid[0]) : 0.0f;
        else out_loss[0] = loss_sum[0];
    }
}
} // namespace xent_impl

// -------------------------
// CLI Helpers
// -------------------------
static inline int arg_int(int argc, char** argv, const char* key, int def) {
    for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == key) return std::atoi(argv[i + 1]);
    return def;
}

int main(int argc, char** argv) {
    const int N = arg_int(argc, argv, "--n", 1024);
    const int C = arg_int(argc, argv, "--c", 1000);
    const int reduction = arg_int(argc, argv, "--reduction", 0); // 0: mean, 1: sum
    const int iters = arg_int(argc, argv, "--iters", 100);

    printf("CrossEntropy FWD: N=%d C=%d reduction=%s iters=%d\n", N, C, (reduction==0?"mean":"sum"), iters);

    float *d_logits, *d_loss, *d_loss_tmp;
    int *d_targets, *d_valid;
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)N * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss_tmp, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid, sizeof(int)));

    // Init (Dummy)
    CUDA_CHECK(cudaMemset(d_logits, 0, (size_t)N * C * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_targets, 0, N * sizeof(int)));

    GpuTimer timer;
    timer.tic();
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaMemsetAsync(d_loss_tmp, 0, sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_valid, 0, sizeof(int)));
        
        xent_impl::xent_fwd_sum_f32<<<N, 256>>>(d_logits, d_targets, N, C, -100, d_loss_tmp, d_valid);
        xent_impl::xent_finalize_loss_f32<<<1, 32>>>(d_loss_tmp, d_valid, reduction, d_loss);
    }
    printf("avg_kernel_ms: %f\n", timer.toc() / iters);

    cudaFree(d_logits); cudaFree(d_targets); cudaFree(d_loss); cudaFree(d_loss_tmp); cudaFree(d_valid);
    return 0;
}