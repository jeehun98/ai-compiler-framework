#pragma once

// C ABI entry point for kernel registration.
// Called from python binding once per process.
extern "C" void aicf_cuda_register_all_kernels();
