#pragma once

#include <cublas_v2.h>

// C = A * B on the GPU. All pointers are DEVICE pointers (from cudaMalloc).
// A is M x K, B is K x N, C is M x N, all row-major.

// TODO(Day 1): one thread per output element.
void launch_matmul_naive(const float* dA, const float* dB, float* dC, int M, int N, int K);

// TODO(Day 1): shared-memory tiling.
void launch_matmul_tiled(const float* dA, const float* dB, float* dC, int M, int N, int K);

// Reference: NVIDIA's cuBLAS library. Already implemented; this is the bar to chase.
void launch_matmul_cublas(cublasHandle_t handle, const float* dA, const float* dB, float* dC, int M, int N, int K);
