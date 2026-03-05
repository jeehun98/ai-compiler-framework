#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

inline void cuda_check(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err)
              << " at " << file << ":" << line << std::endl;
    std::exit(1);
  }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)
