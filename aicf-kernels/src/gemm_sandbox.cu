#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// sandbox 환경을 위한 매크로 및 타이머 (add_sandbox 구조 차용)
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

// -------------------------
// kernels (from launcher.cu)
// -------------------------
namespace gemm_impl {

using namespace nvcuda;

// F32 Naive Strided
__global__ void gemm_f32_naive_kernel(
    const float* A, int64_t Ars, int64_t Acs,
    const float* B, int64_t Brs, int64_t Bcs,
    float* C, int64_t Crs, int64_t Ccs,
    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * Ars + k * Acs] * B[k * Brs + col * Bcs];
        }
        C[row * Crs + col * Ccs] = acc;
    }
}

// F16 WMMA Helpers
__device__ __forceinline__ int ceil16(int x) { return (x + 15) & ~15; }

__device__ __forceinline__ void pack_smemA(__half* smem, const __half* A, int64_t rs, int64_t cs, int M, int K, int m0, int k0, int lane) {
    for (int t = lane; t < 256; t += 32) {
        int i = t / 16, j = t % 16;
        if (m0 + i < M && k0 + j < K) smem[i * 16 + j] = A[(m0 + i) * rs + (k0 + j) * cs];
        else smem[i * 16 + j] = __float2half(0.0f);
    }
}

__device__ __forceinline__ void pack_smemB(__half* smem, const __half* B, int64_t rs, int64_t cs, int K, int N, int k0, int n0, int lane) {
    for (int t = lane; t < 256; t += 32) {
        int i = t / 16, j = t % 16;
        if (k0 + i < K && n0 + j < N) smem[j * 16 + i] = B[(k0 + i) * rs + (n0 + j) * cs];
        else smem[j * 16 + i] = __float2half(0.0f);
    }
}

__global__ void gemm_f16_wmma_kernel(
    const __half* A, int64_t Ars, int64_t Acs, int M, int K,
    const __half* B, int64_t Brs, int64_t Bcs, int K_unused, int N,
    __half* C, int64_t Crs, int64_t Ccs, int M_out, int N_out) {
    
    const int lane = threadIdx.x & 31;
    const int m0 = blockIdx.y * 16;
    const int n0 = blockIdx.x * 16;

    __shared__ __half smemA[256];
    __shared__ __half smemB[256];
    __shared__ float  smemC[256];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += 16) {
        pack_smemA(smemA, A, Ars, Acs, M, K, m0, k0, lane);
        pack_smemB(smemB, B, Brs, Bcs, K, N, k0, n0, lane);
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, smemA, 16);
        wmma::load_matrix_sync(b_frag, smemB, 16);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
        __syncthreads();
    }

    wmma::store_matrix_sync(smemC, acc, 16, wmma::mem_row_major);
    __syncthreads();

    for (int t = lane; t < 256; t += 32) {
        int i = t / 16, j = t % 16;
        if (m0 + i < M && n0 + j < N) C[(m0 + i) * Crs + (n0 + j) * Ccs] = __float2half(smemC[i * 16 + j]);
    }
}

} // namespace gemm_impl

// -------------------------
// simple CLI (add_sandbox style)
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
// validation
// -------------------------
void validate(int M, int N, int K, const float* h_A, const float* h_B, const void* d_C, std::string dtype) {
    std::vector<float> ref(M * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += h_A[m * K + k] * h_B[k * N + n];
            ref[m * N + n] = sum;
        }
    }

    float max_err = 0.0f;
    if (dtype == "f32") {
        std::vector<float> out(M * N);
        CUDA_CHECK(cudaMemcpy(out.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < M * N; ++i) max_err = std::max(max_err, std::fabs(ref[i] - out[i]));
    } else {
        std::vector<__half> out_h(M * N);
        CUDA_CHECK(cudaMemcpy(out_h.data(), d_C, M * N * sizeof(__half), cudaMemcpyDeviceToHost));
        for (int i = 0; i < M * N; ++i) max_err = std::max(max_err, std::fabs(ref[i] - __half2float(out_h[i])));
    }
    std::cout << "max_abs_error: " << max_err << "\n";
}

int main(int argc, char** argv) {
    const int M = arg_int(argc, argv, "--m", 512);
    const int N = arg_int(argc, argv, "--n", 512);
    const int K = arg_int(argc, argv, "--k", 512);
    const int iters = arg_int(argc, argv, "--iters", 100);
    const int warmup = arg_int(argc, argv, "--warmup", 10);
    const std::string dtype = arg_str(argc, argv, "--dtype", "f16");

    std::cout << "GEMM: M=" << M << " N=" << N << " K=" << K << " dtype=" << dtype << " iters=" << iters << "\n";

    // Host Init
    std::vector<float> h_A(M * K), h_B(K * N);
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)(rand() % 100) / 100.0f;

    GpuTimer timer;
    if (dtype == "f32") {
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);

        for (int i = 0; i < warmup; ++i) gemm_impl::gemm_f32_naive_kernel<<<grid, block>>>(d_A, K, 1, d_B, N, 1, d_C, N, 1, M, N, K);
        
        timer.tic();
        for (int i = 0; i < iters; ++i) gemm_impl::gemm_f32_naive_kernel<<<grid, block>>>(d_A, K, 1, d_B, N, 1, d_C, N, 1, M, N, K);
        float ms = timer.toc() / iters;
        
        validate(M, N, K, h_A.data(), h_B.data(), d_C, "f32");
        std::cout << "avg_kernel_ms: " << ms << "\n";
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    } else {
        std::vector<__half> h_Ah(M * K), h_Bh(K * N);
        for (int i = 0; i < M * K; ++i) h_Ah[i] = __float2half(h_A[i]);
        for (int i = 0; i < K * N; ++i) h_Bh[i] = __float2half(h_B[i]);

        __half *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_A, h_Ah.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_Bh.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));

        dim3 block(32, 1);
        dim3 grid((N + 15) / 16, (M + 15) / 16);

        for (int i = 0; i < warmup; ++i) gemm_impl::gemm_f16_wmma_kernel<<<grid, block>>>(d_A, K, 1, M, K, d_B, N, 1, K, N, d_C, N, 1, M, N);

        timer.tic();
        for (int i = 0; i < iters; ++i) gemm_impl::gemm_f16_wmma_kernel<<<grid, block>>>(d_A, K, 1, M, K, d_B, N, 1, K, N, d_C, N, 1, M, N);
        float ms = timer.toc() / iters;

        validate(M, N, K, h_A.data(), h_B.data(), d_C, "f16");
        std::cout << "avg_kernel_ms: " << ms << "\n";
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    return 0;
}