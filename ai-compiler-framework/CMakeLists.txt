cmake_minimum_required(VERSION 3.20)
project(ai_compiler_framework LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(AICF_ENABLE_NVTX "Enable NVTX profiling" ON)

# -----------------------------
# CUDA toolkit
# -----------------------------
find_package(CUDAToolkit REQUIRED)

# -----------------------------
# Core library
# -----------------------------
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

    src/backends/cuda/ops/add/launcher.cu
    src/backends/cuda/ops/relu/launcher.cu
    src/backends/cuda/ops/gemm/launcher.cu
    src/backends/cuda/ops/bias_add/launcher.cu
    src/backends/cuda/ops/reduce_sum/launcher.cu  
    src/backends/cuda/ops/mse_grad/launcher.cu
    src/backends/cuda/ops/relu_bwd/launcher.cu
    src/backends/cuda/ops/sgd_step/launcher.cu
    src/backends/cuda/ops/copy/launcher.cu
    src/backends/cuda/ops/grad_zero/launcher.cu



    src/backends/cuda/registry/registry.cpp
    src/backends/cuda/registry/register_all.cpp
)

target_include_directories(aicf PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# NVTX define
if (AICF_ENABLE_NVTX)
  target_compile_definitions(aicf PUBLIC AICF_ENABLE_NVTX=1)
else()
  target_compile_definitions(aicf PUBLIC AICF_ENABLE_NVTX=0)
endif()

# CUDA architectures:
# - prefer user-provided -DCMAKE_CUDA_ARCHITECTURES
# - fallback to 86
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET aicf PROPERTY CUDA_ARCHITECTURES 86)
else()
  set_property(TARGET aicf PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Link CUDA runtime
target_link_libraries(aicf PUBLIC CUDA::cudart)

# NVTX (Windows: NvToolsExt)
# - If CUDA::nvToolsExt exists, use it
# - else, link local NvToolsExt (Windows)
if (TARGET CUDA::nvToolsExt)
  target_link_libraries(aicf PUBLIC CUDA::nvToolsExt)
elseif (WIN32)
  set(NVTOOLSEXT_ROOT "C:/Program Files/NVIDIA Corporation/NvToolsExt")
  target_include_directories(aicf PUBLIC "${NVTOOLSEXT_ROOT}/include")
  target_link_libraries(aicf PUBLIC "${NVTOOLSEXT_ROOT}/lib/x64/nvToolsExt64_1.lib")
endif()

# -----------------------------
# Examples / benches
# -----------------------------
add_executable(aicf_smoke examples/smoke.cpp)
target_link_libraries(aicf_smoke PRIVATE aicf)

# -----------------------------
# PyTorch / pybind11 extensions
# -----------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")

# NOTE:
# If TorchConfig.cmake is not found, set -DTorch_DIR to:
#   <python>/Lib/site-packages/torch/share/cmake/Torch
find_package(Torch REQUIRED)

# torch_python (needed for pybind11 <-> at::Tensor casters on Windows)
get_filename_component(_TORCH_PREFIX "${Torch_DIR}/../../.." ABSOLUTE)
find_library(TORCH_PYTHON_LIBRARY
  NAMES torch_python
  PATHS "${_TORCH_PREFIX}/lib"
  NO_DEFAULT_PATH
)
message(STATUS "TORCH_PYTHON_LIBRARY = ${TORCH_PYTHON_LIBRARY}")

function(aicf_add_op_pybind target_name source_file output_name)
  add_library(${target_name} MODULE ${source_file})
  target_compile_features(${target_name} PRIVATE cxx_std_17)

  # include paths
  target_include_directories(${target_name} PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/python_bindings
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  # link
  target_link_libraries(${target_name} PRIVATE
    aicf
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    ${Python_LIBRARIES}
    CUDA::cudart
  )

  # Ensure the extension builds for the same CUDA arch as core
  if (DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  else()
    set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES 86)
  endif()

  # Windows: .pyd, no "lib" prefix
  set_target_properties(${target_name} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "${output_name}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/aicf_cuda"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/aicf_cuda"
  )

  if (WIN32)
    set_target_properties(${target_name} PROPERTIES SUFFIX ".pyd")
  endif()

  # Torch extension name define (often required on Windows)
  target_compile_definitions(${target_name} PRIVATE TORCH_EXTENSION_NAME=${output_name})

  # Linux-only ABI flag (Windows에서는 적용하지 않음)
  if (NOT WIN32)
    target_compile_definitions(${target_name} PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
  endif()
endfunction()

# Plan A: single unified module "aicf_cuda" exporting op_call()
aicf_add_op_pybind(aicf_cuda_C src/python_bindings/bindings.cpp _C)
