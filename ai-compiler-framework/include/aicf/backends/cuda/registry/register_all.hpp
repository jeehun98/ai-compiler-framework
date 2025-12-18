#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// CUDA backend kernel registry를 채우는 엔트리 포인트
void aicf_cuda_register_all_kernels();

#ifdef __cplusplus
}
#endif
