#!/usr/bin/env python3
import argparse
from pathlib import Path
from textwrap import dedent


# -----------------------------
# helpers
# -----------------------------
def repo_root_from_this_file() -> Path:
    # tools/codegen/new_op.py  -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def write_file(path: Path, content: str, force: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.write_text(content, encoding="utf-8", newline="\n")


def upper1(s: str) -> str:
    return s[:1].upper() + s[1:]


def macro_guard(op: str) -> str:
    return f"AICF_BACKENDS_CUDA_OPS_{op.upper()}_API_HPP"


# -----------------------------
# templates (v0.2 / Plan A)
# -----------------------------
API_HPP_TPL = """\
#pragma once
#include <cstdint>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

namespace aicf::cuda {{

// Public API (v0.2)
// NOTE: keep headers light. For F16, prefer void* API to avoid cuda_fp16 in headers.

aicf::Status {op}_f32(
    const float* a,
    const float* b,
    float* out,
    int N,
    aicf::Stream stream);

aicf::Status {op}_f16(
    const void* a,
    const void* b,
    void* out,
    int N,
    aicf::Stream stream);

}} // namespace aicf::cuda
"""

KERNELS_CUH_TPL = """\
#pragma once
// kernels are defined in launcher.cu for now (v0.2).
// Keep this file for future split (kernels.cuh / kernels.cu) if needed.
"""

LAUNCHER_CU_TPL = """\
// src/backends/cuda/ops/{op}/launcher.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API
#include <aicf/backends/cuda/ops/{op}/api.hpp>

// registry glue
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>

// common shim
#include "aicf/backends/cuda/ops/_common/shim/launch.hpp"
#include "aicf/backends/cuda/ops/_common/shim/status.hpp"
#include "aicf/backends/cuda/ops/_common/shim/validate.hpp"

#include "kernels.cuh"

namespace aicf::cuda {{

// -------------------------
// kernels
// -------------------------
namespace {op}_impl {{

__global__ void {op}_f32_kernel(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ out,
                               int N) {{
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N) out[i] = a[i] + b[i];
}}

__global__ void {op}_f16_kernel(const __half* __restrict__ a,
                               const __half* __restrict__ b,
                               __half* __restrict__ out,
                               int N) {{
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N) {{
    // TODO: replace with op-specific logic
    out[i] = __hadd(a[i], b[i]);
  }}
}}

__global__ void {op}_f16x2_kernel(const __half2* __restrict__ a,
                                 const __half2* __restrict__ b,
                                 __half2* __restrict__ out,
                                 int N2) {{
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N2) {{
    // TODO: replace with op-specific logic
    out[i] = __hadd2(a[i], b[i]);
  }}
}}

}} // namespace {op}_impl

// -------------------------
// public API implementation
// -------------------------
aicf::Status {op}_f32(const float* a,
                     const float* b,
                     float* out,
                     int N,
                     aicf::Stream stream) {{
  if (!a || !b || !out || N <= 0) return aicf::Status::InvalidArgument;

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  constexpr int kThreads = 256;
  const int blocks = (N + kThreads - 1) / kThreads;
  {op}_impl::{op}_f32_kernel<<<blocks, kThreads, 0, s>>>(a, b, out, N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}}

aicf::Status {op}_f16(const void* a,
                     const void* b,
                     void* out,
                     int N,
                     aicf::Stream stream) {{
  if (!a || !b || !out || N <= 0) return aicf::Status::InvalidArgument;

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  constexpr int kThreads = 256;
  const int blocks = (N + kThreads - 1) / kThreads;
  {op}_impl::{op}_f16_kernel<<<blocks, kThreads, 0, s>>>(
      (const __half*)a, (const __half*)b, (__half*)out, N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}}

// -------------------------
// Registry Variants - v0.2 Plan A
// Contract:
//   inputs[0]=A [N], inputs[1]=B [N], outputs[0]=O [N]
//   binding guarantees: CUDA + contiguous
// -------------------------

static size_t {op}_variant_workspace(const TensorDesc*, int, const void*) {{
  return 0;
}}

// ---- F32 variant ----
static inline bool {op}_f32_variant_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {{

  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& O = outputs[0];

  if (!aicf::cuda::shim::is_f32_contig_1d(A)) return false;
  if (!aicf::cuda::shim::is_f32_contig_1d(B)) return false;
  if (!aicf::cuda::shim::is_f32_contig_1d(O)) return false;

  if (!aicf::cuda::shim::same_shape_1d(A, B)) return false;
  if (!aicf::cuda::shim::same_shape_1d(A, O)) return false;

  return (O.shape[0] > 0);
}}

static bool {op}_f32_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {{
  if (!inputs || !outputs) return false;
  return {op}_f32_variant_check(inputs, num_inputs, outputs, num_outputs);
}}

static aicf::Status {op}_f32_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {{

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;
  if (!{op}_f32_variant_check(inputs, num_inputs, outputs, num_outputs)) {{
    return aicf::Status::InvalidArgument;
  }}

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  const int N = static_cast<int>(O.shape[0]);

  constexpr int kThreads = 256;
  const int blocks = (N + kThreads - 1) / kThreads;

  {op}_impl::{op}_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)A.data,
      (const float*)B.data,
      (float*)O.data,
      N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}}

KernelVariant make_{op}_f32_variant() {{
  KernelVariant v{{}};
  v.name = "{op}_f32_naive";
  v.priority = 0;
  v.flags = 0;
  v.launch = {op}_f32_variant_launch;
  v.supported = {op}_f32_variant_supported;
  v.query_workspace = {op}_variant_workspace;
  return v;
}}

// ---- F16 naive variant ----
static inline bool {op}_f16_variant_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {{

  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& O = outputs[0];

  if (!aicf::cuda::shim::is_f16_contig_1d(A)) return false;
  if (!aicf::cuda::shim::is_f16_contig_1d(B)) return false;
  if (!aicf::cuda::shim::is_f16_contig_1d(O)) return false;

  if (!aicf::cuda::shim::same_shape_1d(A, B)) return false;
  if (!aicf::cuda::shim::same_shape_1d(A, O)) return false;

  return (O.shape[0] > 0);
}}

static bool {op}_f16_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {{
  if (!inputs || !outputs) return false;
  return {op}_f16_variant_check(inputs, num_inputs, outputs, num_outputs);
}}

static aicf::Status {op}_f16_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {{

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;
  if (!{op}_f16_variant_check(inputs, num_inputs, outputs, num_outputs)) {{
    return aicf::Status::InvalidArgument;
  }}

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  const int N = static_cast<int>(O.shape[0]);

  constexpr int kThreads = 256;
  const int blocks = (N + kThreads - 1) / kThreads;

  {op}_impl::{op}_f16_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half*)A.data,
      (const __half*)B.data,
      (__half*)O.data,
      N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}}

KernelVariant make_{op}_f16_variant() {{
  KernelVariant v{{}};
  v.name = "{op}_f16_naive";
  v.priority = 0;
  v.flags = 0;
  v.launch = {op}_f16_variant_launch;
  v.supported = {op}_f16_variant_supported;
  v.query_workspace = {op}_variant_workspace;
  return v;
}}

// ---- F16 vec2 (half2) variant ----
static inline bool {op}_f16_vec2_variant_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {{

  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& O = outputs[0];

  if (!aicf::cuda::shim::is_f16_contig_1d(A)) return false;
  if (!aicf::cuda::shim::is_f16_contig_1d(B)) return false;
  if (!aicf::cuda::shim::is_f16_contig_1d(O)) return false;

  if (!aicf::cuda::shim::same_shape_1d(A, B)) return false;
  if (!aicf::cuda::shim::same_shape_1d(A, O)) return false;

  if (!aicf::cuda::shim::is_even_len_1d(O)) return false;

  if (!aicf::cuda::shim::is_aligned_data(A, 4)) return false;
  if (!aicf::cuda::shim::is_aligned_data(B, 4)) return false;
  if (!aicf::cuda::shim::is_aligned_data(O, 4)) return false;

  return true;
}}

static bool {op}_f16_vec2_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {{
  if (!inputs || !outputs) return false;
  return {op}_f16_vec2_variant_check(inputs, num_inputs, outputs, num_outputs);
}}

static aicf::Status {op}_f16_vec2_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {{

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;
  if (!{op}_f16_vec2_variant_check(inputs, num_inputs, outputs, num_outputs)) {{
    return aicf::Status::InvalidArgument;
  }}

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  const int N = static_cast<int>(O.shape[0]);
  const int N2 = N / 2;

  constexpr int kThreads = 256;
  const int blocks = (N2 + kThreads - 1) / kThreads;

  {op}_impl::{op}_f16x2_kernel<<<blocks, kThreads, 0, stream>>>(
      (const __half2*)A.data,
      (const __half2*)B.data,
      (__half2*)O.data,
      N2);

  return aicf::cuda::shim::cuda_last_error_to_status();
}}

KernelVariant make_{op}_f16_vec2_variant() {{
  KernelVariant v{{}};
  v.name = "{op}_f16_vec2_half2";
  v.priority = 10;
  v.flags = 0;
  v.launch = {op}_f16_vec2_variant_launch;
  v.supported = {op}_f16_vec2_variant_supported;
  v.query_workspace = {op}_variant_workspace;
  return v;
}}

}} // namespace aicf::cuda
"""


def main():
    ap = argparse.ArgumentParser(
        description="AICF op scaffold generator (v0.2 / Plan A). Creates api.hpp / launcher.cu / kernels.cuh."
    )
    ap.add_argument("--op", required=True, help="op folder name, e.g. add, relu, gelu")
    ap.add_argument("--kind", required=True, help="OpKind enum name, e.g. EltwiseAdd")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ap.add_argument("--no-f16", action="store_true", help="skip f16 variants")
    ap.add_argument("--no-vec2", action="store_true", help="skip f16 vec2 half2 variant")
    args = ap.parse_args()

    op = args.op.strip()
    kind = args.kind.strip()

    root = repo_root_from_this_file()

    inc_api = root / "include" / "aicf" / "backends" / "cuda" / "ops" / op / "api.hpp"
    src_dir = root / "src" / "backends" / "cuda" / "ops" / op
    src_launcher = src_dir / "launcher.cu"
    src_kernels = src_dir / "kernels.cuh"

    api_txt = API_HPP_TPL.format(op=op)
    kernels_txt = KERNELS_CUH_TPL.format(op=op)

    launcher_txt = LAUNCHER_CU_TPL.format(op=op)

    if args.no_f16:
        # crude removal: keep f32-only by deleting f16 blocks.
        # For v1, simplest is to just generate and let user delete manually.
        pass
    if args.no_vec2:
        # same note: v1 keeps full template; user can delete vec2 parts.
        pass

    write_file(inc_api, api_txt, args.force)
    write_file(src_kernels, kernels_txt, args.force)
    write_file(src_launcher, launcher_txt, args.force)

    print("\n== Generated ==")
    print(f"  {inc_api}")
    print(f"  {src_kernels}")
    print(f"  {src_launcher}")

    print("\n== Next manual steps (v1) ==")
    print("1) Add factories to include/aicf/backends/cuda/registry/register_all.hpp:")
    if not args.no_f16 and not args.no_vec2:
        print(f"   KernelVariant make_{op}_f32_variant();")
        print(f"   KernelVariant make_{op}_f16_variant();")
        print(f"   KernelVariant make_{op}_f16_vec2_variant();")
    elif not args.no_f16 and args.no_vec2:
        print(f"   KernelVariant make_{op}_f32_variant();")
        print(f"   KernelVariant make_{op}_f16_variant();")
    else:
        print(f"   KernelVariant make_{op}_f32_variant();")

    print("\n2) Register variants in src/backends/cuda/registry/register_all.cpp:")
    print(f"   // {kind}")
    print(f"   R.register_kernel(OpKind::{kind}, make_{op}_f32_variant());")
    if not args.no_f16:
        print(f"   R.register_kernel(OpKind::{kind}, make_{op}_f16_variant());")
        if not args.no_vec2:
            print(f"   R.register_kernel(OpKind::{kind}, make_{op}_f16_vec2_variant());")

    print("\n3) Add OpKind enum value in include/aicf/backends/cuda/registry/op_kind.hpp:")
    print(f"   {kind} = <next_id>,")

    print("\n4) Add source to build if your CMakeLists doesn't glob:")
    print(f"   src/backends/cuda/ops/{op}/launcher.cu")

    print("\n5) Implement correct math inside kernels (template currently uses add).")
    print("Done.")


if __name__ == "__main__":
    main()
