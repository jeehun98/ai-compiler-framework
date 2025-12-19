#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "common.hpp"
#include <aicf/backends/cuda/ops/gemm/api.hpp>

namespace py = pybind11;

PYBIND11_MODULE(gemm, m) {
    m.doc() = "AICF CUDA gemm ops";

    m.def("gemm_f32", [](const torch::Tensor& A,
                         const torch::Tensor& B,
                         torch::Tensor& C) -> bool {
        if (!is_cuda_f32_contig(A) || !is_cuda_f32_contig(B) || !is_cuda_f32_contig(C)) return false;
        if (A.dim() != 2 || B.dim() != 2 || C.dim() != 2) return false;

        const int M = (int)A.size(0);
        const int K = (int)A.size(1);
        const int K2 = (int)B.size(0);
        const int N = (int)B.size(1);
        if (K2 != K) return false;
        if ((int)C.size(0) != M || (int)C.size(1) != N) return false;

        auto st = aicf::cuda::gemm_f32(
            (const float*)A.data_ptr<float>(),
            (const float*)B.data_ptr<float>(),
            (float*)C.data_ptr<float>(),
            M, N, K, default_stream());

        return status_ok(st);
    }, py::arg("A"), py::arg("B"), py::arg("C"));
}
