// aicf-kernels/src/add_sandbox.cu
// - AICF ops/add의 분기 로직을 샌드박스로 이식한 버전
// - f32 naive / f16 naive / f16 half2(vec2) fastpath
// - 실행: build\bin\add_sandbox.exe --dtype f16 --n 16777216 --iters 200
// - 프로파일: scripts\ncu_fast.bat build\bin\add_sandbox.exe --dtype f16 --n 16777216 --iters 10

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../common/cuda_check.cuh"
#include "../common/timer.cuh"

// -------------------------
// pointer alignment helper
// -------------------------
static inline bool is_aligned_ptr(const void* p, size_t align) {
  return (reinterpret_cast<uintptr_t>(p) % reinterpret_cast<uintptr_t>(align)) == 0;
}

// -------------------------
// kernels (same as AICF)
// -------------------------
namespace add_impl {

__global__ void add_f32_kernel(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ out,
                               int N) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N) out[i] = a[i] + b[i];
}

__global__ void add_f16_kernel(const __half* __restrict__ a,
                               const __half* __restrict__ b,
                               __half* __restrict__ out,
                               int N) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N) out[i] = __hadd(a[i], b[i]);
}

__global__ void add_f16x2_kernel(const __half2* __restrict__ a,
                                 const __half2* __restrict__ b,
                                 __half2* __restrict__ out,
                                 int N2) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N2) out[i] = __hadd2(a[i], b[i]);
}

}  // namespace add_impl

// -------------------------
// simple CLI
// -------------------------
static inline int arg_int(int argc, char** argv, const char* key, int def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == key) return std::atoi(argv[i + 1]);
  }
  return def;
}

static inline std::string arg_str(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == key) return std::string(argv[i + 1]);
  }
  return std::string(def);
}

// -------------------------
// validation
// -------------------------
static inline void validate_f32(const std::vector<float>& ref, const std::vector<float>& out) {
  float max_err = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) {
    max_err = std::max(max_err, std::fabs(ref[i] - out[i]));
  }
  std::cout << "max_abs_error: " << max_err << "\n";
}

static inline void validate_f16_against_f32ref(const std::vector<float>& ref,
                                               const std::vector<__half>& out_h) {
  float max_err = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) {
    float out = __half2float(out_h[i]);
    max_err = std::max(max_err, std::fabs(ref[i] - out));
  }
  std::cout << "max_abs_error: " << max_err << "\n";
}

// -------------------------
// host init
// -------------------------
static inline void fill_f32(std::vector<float>& a, std::vector<float>& b) {
  // deterministic-ish
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = (float)(i % 1024) * 0.001f;
    b[i] = (float)((i * 7) % 1024) * 0.001f;
  }
}

static inline void f32_to_f16(const std::vector<float>& in, std::vector<__half>& out) {
  for (size_t i = 0; i < in.size(); ++i) out[i] = __float2half(in[i]);
}

// -------------------------
// launcher (mirrors AICF branching)
// -------------------------
struct LaunchInfo {
  enum class Path { F32, F16_NAIVE, F16_HALF2 } path;
};

static inline LaunchInfo choose_path(const std::string& dtype, int64_t N,
                                     const void* a, const void* b, const void* out) {
  if (dtype == "f32") return {LaunchInfo::Path::F32};

  // f16
  bool can_half2 = ((N & 1) == 0) &&
                   is_aligned_ptr(a, 4) &&
                   is_aligned_ptr(b, 4) &&
                   is_aligned_ptr(out, 4);
  if (can_half2) return {LaunchInfo::Path::F16_HALF2};
  return {LaunchInfo::Path::F16_NAIVE};
}

int main(int argc, char** argv) {
  const int n = arg_int(argc, argv, "--n", 1 << 24);
  const int iters = arg_int(argc, argv, "--iters", 200);
  const std::string dtype = arg_str(argc, argv, "--dtype", "f16");  // f16 or f32
  const int warmup = arg_int(argc, argv, "--warmup", 10);

  if (!(dtype == "f16" || dtype == "f32")) {
    std::cerr << "Unsupported --dtype. Use f16 or f32.\n";
    return 2;
  }
  if (n <= 0) {
    std::cerr << "--n must be > 0\n";
    return 2;
  }

  std::cout << "dtype=" << dtype << " n=" << n << " iters=" << iters << " warmup=" << warmup << "\n";

  constexpr int kThreads = 256;

  if (dtype == "f32") {
    // ---- host ----
    std::vector<float> h_a(n), h_b(n), h_out(n), h_ref(n);
    fill_f32(h_a, h_b);
    for (int i = 0; i < n; ++i) h_ref[i] = h_a[i] + h_b[i];

    // ---- device ----
    float *d_a=nullptr, *d_b=nullptr, *d_out=nullptr;
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    int blocks = (n + kThreads - 1) / kThreads;

    // warmup
    for (int i = 0; i < warmup; ++i)
      add_impl::add_f32_kernel<<<blocks, kThreads>>>(d_a, d_b, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer t;
    t.tic();
    for (int i = 0; i < iters; ++i)
      add_impl::add_f32_kernel<<<blocks, kThreads>>>(d_a, d_b, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = t.toc() / (float)iters;

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    validate_f32(h_ref, h_out);

    std::cout << "avg_kernel_ms: " << ms << "\n";

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return 0;
  }

  // ---- f16 path ----
  // host: keep f32 ref for validation
  std::vector<float> h_a_f32(n), h_b_f32(n), h_ref_f32(n);
  fill_f32(h_a_f32, h_b_f32);
  for (int i = 0; i < n; ++i) h_ref_f32[i] = h_a_f32[i] + h_b_f32[i];

  std::vector<__half> h_a(n), h_b(n), h_out(n);
  f32_to_f16(h_a_f32, h_a);
  f32_to_f16(h_b_f32, h_b);

  // device
  __half *d_a=nullptr, *d_b=nullptr, *d_out=nullptr;
  size_t bytes = (size_t)n * sizeof(__half);
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  LaunchInfo info = choose_path(dtype, (int64_t)n, d_a, d_b, d_out);

  // launch
  if (info.path == LaunchInfo::Path::F16_HALF2) {
    int N2 = n / 2;
    int blocks = (N2 + kThreads - 1) / kThreads;

    std::cout << "path=f16_half2\n";

    for (int i = 0; i < warmup; ++i)
      add_impl::add_f16x2_kernel<<<blocks, kThreads>>>(
          (const __half2*)d_a, (const __half2*)d_b, (__half2*)d_out, N2);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer t;
    t.tic();
    for (int i = 0; i < iters; ++i)
      add_impl::add_f16x2_kernel<<<blocks, kThreads>>>(
          (const __half2*)d_a, (const __half2*)d_b, (__half2*)d_out, N2);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = t.toc() / (float)iters;

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    validate_f16_against_f32ref(h_ref_f32, h_out);

    std::cout << "avg_kernel_ms: " << ms << "\n";
  } else {
    int blocks = (n + kThreads - 1) / kThreads;

    std::cout << "path=f16_naive\n";

    for (int i = 0; i < warmup; ++i)
      add_impl::add_f16_kernel<<<blocks, kThreads>>>(d_a, d_b, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer t;
    t.tic();
    for (int i = 0; i < iters; ++i)
      add_impl::add_f16_kernel<<<blocks, kThreads>>>(d_a, d_b, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = t.toc() / (float)iters;

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    validate_f16_against_f32ref(h_ref_f32, h_out);

    std::cout << "avg_kernel_ms: " << ms << "\n";
  }

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
  return 0;
}