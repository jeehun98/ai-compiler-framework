cmake_minimum_required(VERSION 3.20)
project(AICF_Kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# If user doesn't set it, default to 86 (Ampere GA10x) — change as needed.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
include_directories(common)

# Reasonable default flags: fast builds + lineinfo for profiling.
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G>)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
endif()

file(GLOB KERNEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
foreach(FILE ${KERNEL_FILES})
  get_filename_component(NAME ${FILE} NAME_WE)
  add_executable(${NAME} ${FILE})
endforeach()
