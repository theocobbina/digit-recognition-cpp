#pragma once

// C = A * B, where A is M x K, B is K x N, C is M x N (all row-major).

// Baseline: the textbook triple loop (i, j, k). Already implemented.
void matmul_cpu_naive(const float* A, const float* B, float* C, int M, int N, int K);

// TODO(Day 1): your faster CPU version. See src/cpu/matmul_cpu.cpp for hints.
void matmul_cpu_reordered(const float* A, const float* B, float* C, int M, int N, int K);
