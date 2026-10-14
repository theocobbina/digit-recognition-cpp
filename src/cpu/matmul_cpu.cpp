#include "matmul_cpu.h"

void matmul_cpu_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // B[k * N + j] jumps N floats every step -> poor cache use.
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_cpu_reordered(const float* A, const float* B, float* C, int M, int N, int K) {
    // Same math as the naive version, loops in (i, k, j) order instead of (i, j, k).
    // The inner loop now walks ACROSS rows of B and C, which sit next to each other
    // in memory, so every cache line fetched (64 bytes = 16 floats) is fully used.
    //
    // Interview question to be ready for: why is (i, k, j) faster than (i, j, k)
    // when both do exactly the same number of multiplies?
    for (int idx = 0; idx < M * N; ++idx) C[idx] = 0.0f;  // we accumulate with +=

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a = A[i * K + k];  // A[i][k], read once and reused across the row
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}
