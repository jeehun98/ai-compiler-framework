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
// AdamStep Kernels (Fused m, v, param update)
// ============================================================================
namespace adam_impl {

__global__ void adam_f32_kernel(
    float* __restrict__ param,      // theta
    float* __restrict__ m,          // 1st moment
    float* __restrict__ v,          // 2nd moment
    const float* __restrict__ grad, // gradient
    float lr, float beta1, float beta2, float eps,
    float bias_correction1, float bias_correction2,
    int64_t n) {
    
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];
    float m_t = m[i];
    float v_t = v[i];

    // 1. Update moments
    m_t = beta1 * m_t + (1.0f - beta1) * g;
    v_t = beta2 * v_t + (1.0f - beta2) * g * g;

    // 2. Bias correction & update
    float m_hat = m_t / bias_correction1;
    float v_hat = v_t / bias_correction2;

    param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);

    // 3. Write back
    m[i] = m_t;
    v[i] = v_t;
}

// F16 Mixed Precision: 연산은 f32로 수행하여 수치적 안정성 확보
__global__ void adam_f16_kernel(
    __half* __restrict__ param,
    __half* __restrict__ m,
    __half* __restrict__ v,
    const __half* __restrict__ grad,
    float lr, float beta1, float beta2, float eps,
    float bias_correction1, float bias_correction2,
    int64_t n) {

    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = __half2float(grad[i]);
    float m_t = __half2float(m[i]);
    float v_t = __half2float(v[i]);

    m_t = beta1 * m_t + (1.0f - beta1) * g;
    v_t = beta2 * v_t + (1.0f - beta2) * g * g;

    float m_hat = m_t / bias_correction1;
    float v_hat = v_t / bias_correction2;

    float p = __half2float(param[i]);
    p -= lr * m_hat / (sqrtf(v_hat) + eps);

    param[i] = __float2half(p);
    m[i] = __float2half(m_t);
    v[i] = __float2half(v_t);
}

} // namespace adam_impl

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

int main(int argc, char** argv) {
    const int64_t n = arg_int(argc, argv, "--n", 1 << 20); // 약 1M 원소
    const int iters = arg_int(argc, argv, "--iters", 100);
    const std::string dtype = arg_str(argc, argv, "--dtype", "f32");

    printf("AdamStep: n=%lld dtype=%s iters=%d\n", n, dtype.c_str(), iters);

    // Adam Hyperparameters
    float lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f, bc2 = 1.0f; // Simplified bias correction for benchmark

    size_t sz = (dtype == "f32") ? n * sizeof(float) : n * sizeof(__half);
    void *d_p, *d_m, *d_v, *d_g;
    CUDA_CHECK(cudaMalloc(&d_p, sz));
    CUDA_CHECK(cudaMalloc(&d_m, sz));
    CUDA_CHECK(cudaMalloc(&d_v, sz));
    CUDA_CHECK(cudaMalloc(&d_g, sz));
    CUDA_CHECK(cudaMemset(d_m, 0, sz));
    CUDA_CHECK(cudaMemset(d_v, 0, sz));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    GpuTimer timer;
    if (dtype == "f32") {
        timer.tic();
        for (int i = 0; i < iters; ++i) {
            adam_impl::adam_f32_kernel<<<grid, block>>>((float*)d_p, (float*)d_m, (float*)d_v, (const float*)d_g, lr, beta1, beta2, eps, bc1, bc2, n);
        }
        printf("avg_kernel_ms: %f\n", timer.toc() / iters);
    } else {
        timer.tic();
        for (int i = 0; i < iters; ++i) {
            adam_impl::adam_f16_kernel<<<grid, block>>>((__half*)d_p, (__half*)d_m, (__half*)d_v, (const __half*)d_g, lr, beta1, beta2, eps, bc1, bc2, n);
        }
        printf("avg_kernel_ms: %f\n", timer.toc() / iters);
    }

    cudaFree(d_p); cudaFree(d_m); cudaFree(d_v); cudaFree(d_g);
    return 0;
}