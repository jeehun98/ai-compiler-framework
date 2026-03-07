#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

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
// BatchNorm Kernels (From launcher.cu)
// ============================================================================
namespace bn_impl {

__global__ void bn_fwd_stats_f16_atomic(const __half* __restrict__ x, float* __restrict__ sum, float* __restrict__ sumsq, int N, int C, int HW) {
    const int64_t total = (int64_t)N * C * HW;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
        const int c = (int)((i / HW) % C);
        const float v = __half2float(x[i]);
        atomicAdd(&sum[c], v);
        atomicAdd(&sumsq[c], v * v);
    }
}

__global__ void bn_finalize_mean_var(float* __restrict__ mean, float* __restrict__ var, int C, float invNHW) {
    const int c = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float m = mean[c] * invNHW;
    float v = var[c] * invNHW - m * m;
    mean[c] = m;
    var[c] = fmaxf(v, 0.0f);
}

__global__ void bn_var_to_rstd_inplace(float* __restrict__ var_inplace, int C, float eps) {
    const int c = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    var_inplace[c] = rsqrtf(var_inplace[c] + eps);
}

__global__ void bn_fwd_apply_f16(const __half* __restrict__ x, const __half* __restrict__ gamma, const __half* __restrict__ beta,
                                 const float* __restrict__ mean, const float* __restrict__ rstd, __half* __restrict__ y, int N, int C, int HW) {
    const int64_t total = (int64_t)N * C * HW;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
        const int c = (int)((i / HW) % C);
        float mu = mean[c], rs = rstd[c];
        float g = gamma ? __half2float(gamma[c]) : 1.0f;
        float b = beta ? __half2float(beta[c]) : 0.0f;
        y[i] = __float2half_rn(((__half2float(x[i]) - mu) * rs) * g + b);
    }
}
} // namespace bn_impl

// -------------------------
// CLI Helpers
// -------------------------
static inline int arg_int(int argc, char** argv, const char* key, int def) {
    for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == key) return std::atoi(argv[i + 1]);
    return def;
}

int main(int argc, char** argv) {
    const int N = arg_int(argc, argv, "--n", 16);
    const int C = arg_int(argc, argv, "--c", 64);
    const int H = arg_int(argc, argv, "--h", 56);
    const int W = arg_int(argc, argv, "--w", 56);
    const int iters = arg_int(argc, argv, "--iters", 100);
    const float eps = 1e-5f;

    const int HW = H * W;
    const int64_t numel = (int64_t)N * C * HW;

    printf("BatchNorm FWD (Training): N=%d C=%d H=%d W=%d, total=%lld, iters=%d\n", N, C, H, W, numel, iters);

    // Host/Device Memory
    __half *d_x, *d_y, *d_gamma, *d_beta;
    float *d_mean, *d_rstd;
    CUDA_CHECK(cudaMalloc(&d_x, numel * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_y, numel * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_gamma, C * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_beta, C * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_mean, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rstd, C * sizeof(float)));

    // Init
    CUDA_CHECK(cudaMemset(d_x, 0, numel * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_gamma, 0, C * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_beta, 0, C * sizeof(__half)));

    GpuTimer timer;
    const int threads = 256;
    const int blocks = (int)std::min((int64_t)4096, (numel + threads - 1) / threads);
    const int blocksC = (C + threads - 1) / threads;

    timer.tic();
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaMemsetAsync(d_mean, 0, C * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_rstd, 0, C * sizeof(float)));
        
        bn_impl::bn_fwd_stats_f16_atomic<<<blocks, threads>>>(d_x, d_mean, d_rstd, N, C, HW);
        bn_impl::bn_finalize_mean_var<<<blocksC, threads>>>(d_mean, d_rstd, C, 1.0f / (N * HW));
        bn_impl::bn_var_to_rstd_inplace<<<blocksC, threads>>>(d_rstd, C, eps);
        bn_impl::bn_fwd_apply_f16<<<blocks, threads>>>(d_x, d_gamma, d_beta, d_mean, d_rstd, d_y, N, C, HW);
    }
    
    printf("avg_kernel_ms: %f\n", timer.toc() / iters);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma); cudaFree(d_beta);
    cudaFree(d_mean); cudaFree(d_rstd);
    return 0;
}