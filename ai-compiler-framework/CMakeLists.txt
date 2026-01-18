cmake_minimum_required(VERSION 3.20)
project(ai_compiler_framework LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(AICF_ENABLE_NVTX "Enable NVTX profiling" ON)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(CUDAToolkit REQUIRED)

# -----------------------------
# Core library (aicf): CUDA backend only
# -----------------------------
add_library(aicf


  # ops
  src/backends/cuda/ops/reduce_sum/launcher.cu

  
  #src/backends/cuda/ops/add/launcher.cu
  #src/backends/cuda/ops/relu/launcher.cu
  #src/backends/cuda/ops/gemm/launcher.cu
  #src/backends/cuda/ops/bias_add/launcher.cu
  #src/backends/cuda/ops/mse_grad/launcher.cu
  #src/backends/cuda/ops/relu_bwd/launcher.cu
  #src/backends/cuda/ops/sgd_step/launcher.cu
  #src/backends/cuda/ops/copy/launcher.cu
  #src/backends/cuda/ops/grad_zero/launcher.cu
  #src/backends/cuda/ops/adam_step/launcher.cu
  #src/backends/cuda/ops/step_inc/launcher.cu
  #src/backends/cuda/ops/biascorr/launcher.cu
  #src/backends/cuda/ops/layernorm/launcher.cu
  #src/backends/cuda/ops/batchnorm/launcher.cu

  # registry
  src/backends/cuda/registry/registry.cpp
  src/backends/cuda/registry/register_all.cpp
)

target_include_directories(aicf PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(aicf PUBLIC CUDA::cudart)

if (AICF_ENABLE_NVTX)
  target_compile_definitions(aicf PUBLIC AICF_ENABLE_NVTX=1)
else()
  target_compile_definitions(aicf PUBLIC AICF_ENABLE_NVTX=0)
endif()

if (TARGET CUDA::nvToolsExt)
  target_link_libraries(aicf PUBLIC CUDA::nvToolsExt)
elseif (WIN32)
  set(NVTOOLSEXT_ROOT "C:/Program Files/NVIDIA Corporation/NvToolsExt")
  target_include_directories(aicf PUBLIC "${NVTOOLSEXT_ROOT}/include")
  target_link_libraries(aicf PUBLIC "${NVTOOLSEXT_ROOT}/lib/x64/nvToolsExt64_1.lib")
endif()

# -----------------------------
# PyTorch extension: single module _C
# -----------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)

get_filename_component(_TORCH_PREFIX "${Torch_DIR}/../../.." ABSOLUTE)
find_library(TORCH_PYTHON_LIBRARY
  NAMES torch_python
  PATHS "${_TORCH_PREFIX}/lib"
  NO_DEFAULT_PATH
)

add_library(aicf_cuda_C MODULE src/python_bindings/bindings.cpp)

target_include_directories(aicf_cuda_C PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${Python_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src/python_bindings  # common.hpp 쓸 때만
)

target_link_libraries(aicf_cuda_C PRIVATE
  aicf
  ${TORCH_LIBRARIES}
  ${Python_LIBRARIES}
  CUDA::cudart
)

if (TORCH_PYTHON_LIBRARY)
  target_link_libraries(aicf_cuda_C PRIVATE ${TORCH_PYTHON_LIBRARY})
endif()

set_target_properties(aicf_cuda_C PROPERTIES
  PREFIX ""
  OUTPUT_NAME "_C"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/aicf_cuda"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/aicf_cuda"
)

if (WIN32)
  set_target_properties(aicf_cuda_C PROPERTIES SUFFIX ".pyd")
endif()

target_compile_definitions(aicf_cuda_C PRIVATE TORCH_EXTENSION_NAME=_C)
