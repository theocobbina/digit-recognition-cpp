#pragma once

#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Wrap every CUDA call: GPU errors are silent unless you check for them.
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                         cudaGetErrorString(err_), __FILE__, __LINE__);           \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t st_ = (call);                                              \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",                    \
                         static_cast<int>(st_), __FILE__, __LINE__);              \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)
