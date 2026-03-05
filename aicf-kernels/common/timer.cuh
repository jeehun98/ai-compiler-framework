#pragma once
#include <cuda_runtime.h>

struct GpuTimer {
  cudaEvent_t start{}, stop{};
  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }
  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void tic(cudaStream_t s = 0) { cudaEventRecord(start, s); }
  float toc(cudaStream_t s = 0) {
    cudaEventRecord(stop, s);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
  }
};
