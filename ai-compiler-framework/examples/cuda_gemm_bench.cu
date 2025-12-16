#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

#include <aicf/core/status.hpp>
#include <aicf/core/log.hpp>
#include <aicf/runtime/graph.hpp>
#include <aicf/runtime/stream.hpp>
#include <aicf/backends/cuda/ops/gemm/api.hpp>

static void check_cuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

static void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = acc;
        }
    }
}

int main() {
    aicf::init_runtime();

    int M = 64, N = 64, K = 64;
    std::vector<float> hA(M*K), hB(K*N), hC(M*N, 0.f), hRef(M*N, 0.f);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    cpu_gemm(hA.data(), hB.data(), hRef.data(), M, N, K);

    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    check_cuda(cudaMalloc(&dA, sizeof(float)*hA.size()), "malloc dA");
    check_cuda(cudaMalloc(&dB, sizeof(float)*hB.size()), "malloc dB");
    check_cuda(cudaMalloc(&dC, sizeof(float)*hC.size()), "malloc dC");

    check_cuda(cudaMemcpy(dA, hA.data(), sizeof(float)*hA.size(), cudaMemcpyHostToDevice), "H2D A");
    check_cuda(cudaMemcpy(dB, hB.data(), sizeof(float)*hB.size(), cudaMemcpyHostToDevice), "H2D B");
    check_cuda(cudaMemset(dC, 0, sizeof(float)*hC.size()), "memset C");

    cudaStream_t s;
    check_cuda(cudaStreamCreate(&s), "stream create");

    aicf::Stream stream;
    stream.handle = s;

    aicf::Status st = aicf::cuda::gemm_f32(dA, dB, dC, M, N, K, stream);
    if (st != aicf::Status::Ok) {
        std::fprintf(stderr, "gemm_f32 failed: %s\n", aicf::status_to_string(st));
        return 1;
    }

    check_cuda(cudaStreamSynchronize(s), "stream sync");
    check_cuda(cudaMemcpy(hC.data(), dC, sizeof(float)*hC.size(), cudaMemcpyDeviceToHost), "D2H C");

    // verify
    float max_abs = 0.f;
    for (int i = 0; i < M*N; ++i) {
        float diff = std::fabs(hC[i] - hRef[i]);
        if (diff > max_abs) max_abs = diff;
    }

    std::printf("[OK] GEMM f32 naive | M=%d N=%d K=%d | max_abs_diff=%.6g\n", M, N, K, max_abs);

    check_cuda(cudaStreamDestroy(s), "stream destroy");
    check_cuda(cudaFree(dA), "free dA");
    check_cuda(cudaFree(dB), "free dB");
    check_cuda(cudaFree(dC), "free dC");

    aicf::shutdown_runtime();
    return 0;
}
