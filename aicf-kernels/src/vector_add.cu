#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

#include "../common/cuda_check.cuh"
#include "../common/timer.cuh"
#include "../common/validate.cuh"

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < n) c[idx] = a[idx] + b[idx];
}

int main(int argc, char** argv) {
  int n = 1 << 24;
  int iters = 200;

  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--n" && i + 1 < argc) n = std::atoi(argv[++i]);
    else if (k == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
  }

  size_t bytes = (size_t)n * sizeof(float);
  std::vector<float> h_a(n), h_b(n), h_c(n), h_ref(n);

  for (int i = 0; i < n; ++i) {
    h_a[i] = i * 0.5f;
    h_b[i] = i * 0.25f;
    h_ref[i] = h_a[i] + h_b[i];
  }

  float *d_a=nullptr, *d_b=nullptr, *d_c=nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n + block - 1) / block;

  // Warmup
  for (int i = 0; i < 10; ++i) vector_add<<<grid, block>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer t;
  t.tic();
  for (int i = 0; i < iters; ++i) vector_add<<<grid, block>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = t.toc() / (float)iters;

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  validate_max_abs(h_ref.data(), h_c.data(), n);

  std::cout << "avg_kernel_ms: " << ms << std::endl;

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}
