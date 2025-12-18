#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#include <aicf/backends/cuda/nvtx.hpp>
#include <aicf/backends/cuda/ops/eltwise/api.hpp>

#define CHECK_CUDA(x) do {                           \
  cudaError_t e = (x);                               \
  if (e != cudaSuccess) {                            \
    printf("CUDA error %s:%d: %s\n",                 \
           __FILE__, __LINE__, cudaGetErrorString(e)); \
    return 1;                                        \
  }                                                  \
} while(0)

#if __has_include(<nvtx3/nvToolsExt.h>)
  #include <nvtx3/nvToolsExt.h>
#else
  #include <nvToolsExt.h>
#endif

struct NvtxStartEnd {
  nvtxRangeId_t id;
  explicit NvtxStartEnd(const char* msg) : id(nvtxRangeStartA(msg)) {}
  ~NvtxStartEnd() { nvtxRangeEnd(id); }
  NvtxStartEnd(const NvtxStartEnd&) = delete;
  NvtxStartEnd& operator=(const NvtxStartEnd&) = delete;
};

int main() {

  #if AICF_ENABLE_NVTX
    printf("[INFO] NVTX ENABLED\n");
  #else
    printf("[INFO] NVTX DISABLED\n");
  #endif

  constexpr int64_t N = 1 << 20;

  std::vector<float> h_a(N), h_b(N), h_out(N);

  for (int64_t i = 0; i < N; ++i) {
    h_a[i] = float(i) * 0.1f;
    h_b[i] = float(i) * 0.2f - 10.f;
  }

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  cudaStream_t stream = cudaStreamPerThread;


  {
    NvtxStartEnd r("aicf::eltwise::add_f32");
    aicf::cuda::ops::eltwise::add_f32(d_out, d_a, d_b, N, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }
  {
    NvtxStartEnd r("aicf::eltwise::relu_f32");
    aicf::cuda::ops::eltwise::relu_f32(d_out, d_out, N, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }


  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

  float max_diff = 0.f;
  for (int64_t i = 0; i < N; ++i) {
    float ref = std::max(h_a[i] + h_b[i], 0.f);
    max_diff = std::max(max_diff, std::abs(h_out[i] - ref));
  }

  printf("[OK] eltwise add+relu | N=%lld | max_abs_diff=%.6e\n",
         (long long)N, max_diff);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  return 0;
}
