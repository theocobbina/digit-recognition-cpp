#pragma once

#include <cmath>
#include <random>
#include <vector>

// Row-major matrix: element (r, c) lives at data[r * cols + c].
// Every kernel in this project (CPU and GPU) uses this same layout.
struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<float> data;

    Matrix() = default;
    Matrix(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * c, 0.0f) {}

    float& at(int r, int c) { return data[static_cast<size_t>(r) * cols + c]; }
    float at(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }
    size_t size() const { return data.size(); }
};

// Fill with uniform random values in [-1, 1]. Fixed seed so runs are reproducible.
inline void fill_random(Matrix& m, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : m.data) x = dist(rng);
}

// Largest element-wise difference. Used to check a fast kernel against a trusted one.
inline float max_abs_diff(const Matrix& a, const Matrix& b) {
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) worst = std::fmax(worst, std::fabs(a.data[i] - b.data[i]));
    return worst;
}

// Matrix multiply does 2*M*N*K floating-point operations (one multiply + one add per term).
inline double gflops(int M, int N, int K, double ms) {
    return (2.0 * M * N * K) / (ms * 1e6);
}
