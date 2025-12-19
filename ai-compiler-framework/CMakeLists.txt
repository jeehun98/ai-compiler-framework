cmake_minimum_required(VERSION 3.20)
project(ai_compiler_framework LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(AICF_ENABLE_NVTX "Enable NVTX profiling" ON)

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

    src/backends/cuda/registry/registry.cpp
    src/backends/cuda/registry/register_all.cpp
)

target_include_directories(aicf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# NVTX (Windows)
if (WIN32)
  set(NVTOOLSEXT_ROOT "C:/Program Files/NVIDIA Corporation/NvToolsExt")

  target_include_directories(aicf PUBLIC
    "${NVTOOLSEXT_ROOT}/include"
  )

  target_link_libraries(aicf PUBLIC
    "${NVTOOLSEXT_ROOT}/lib/x64/nvToolsExt64_1.lib"
  )
endif()

if (AICF_ENABLE_NVTX)
    target_compile_definitions(aicf PUBLIC AICF_ENABLE_NVTX=1)
else()
    target_compile_definitions(aicf PUBLIC AICF_ENABLE_NVTX=0)
endif()

# Ampere(sm_86) 고정
set_target_properties(aicf PROPERTIES CUDA_ARCHITECTURES "86")

# CUDA toolkit + nvToolsExt (CMake package)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(aicf PUBLIC CUDA::nvToolsExt)

# -----------------------------
# Examples / benches
# -----------------------------
add_executable(aicf_smoke examples/smoke.cpp)
target_link_libraries(aicf_smoke PRIVATE aicf)

add_executable(aicf_gemm_bench examples/cuda_gemm_bench.cu)
target_link_libraries(aicf_gemm_bench PRIVATE aicf)

add_executable(aicf_eltwise_bench examples/cuda_eltwise_bench.cu)
target_link_libraries(aicf_eltwise_bench PRIVATE aicf)

# -----------------------------
# PyTorch / pybind11 extensions
# -----------------------------
# Python Development headers/libs are REQUIRED for Python.h
find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")

# NOTE: If TorchConfig.cmake is not found, set CMAKE_PREFIX_PATH to:
#   <python>/Lib/site-packages/torch/share/cmake
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

  target_link_libraries(${target_name} PRIVATE
    aicf
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    ${Python_LIBRARIES}
  )


  target_include_directories(${target_name} PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/python_bindings
  )

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
endfunction()

aicf_add_op_pybind(aicf_cuda_add  src/python_bindings/add.cpp  add)
aicf_add_op_pybind(aicf_cuda_relu src/python_bindings/relu.cpp relu)
aicf_add_op_pybind(aicf_cuda_gemm src/python_bindings/gemm.cpp gemm)
