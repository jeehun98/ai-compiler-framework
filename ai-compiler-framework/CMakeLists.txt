cmake_minimum_required(VERSION 3.20)
project(ai_compiler_framework LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(AICF_ENABLE_NVTX "Enable NVTX profiling" ON)

add_library(aicf
    src/core/status.cpp
    src/core/log.cpp
    src/runtime/graph.cpp
    src/runtime/capture_plan.cpp
    src/runtime/graph_exec.cpp
    src/ir/ir.cpp
    src/ir/lowering.cpp
    src/backends/cuda/cuda_context.cu
    src/backends/cuda/nvtx.cu
    src/backends/cuda/ops/gemm/launcher.cu

)

target_include_directories(aicf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (AICF_ENABLE_NVTX)
    target_compile_definitions(aicf PUBLIC AICF_ENABLE_NVTX=1)
else()
    target_compile_definitions(aicf PUBLIC AICF_ENABLE_NVTX=0)
endif()

# 너 GPU가 Ampere(sm_86) 기준이라 일단 86으로 고정
set_target_properties(aicf PROPERTIES CUDA_ARCHITECTURES "86")

add_executable(aicf_smoke examples/smoke.cpp)
target_link_libraries(aicf_smoke PRIVATE aicf)

add_executable(aicf_gemm_bench examples/cuda_gemm_bench.cu)
target_link_libraries(aicf_gemm_bench PRIVATE aicf)
