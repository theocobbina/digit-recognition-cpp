// CPU matrix-multiply benchmark.
// Times each implementation, checks it against the naive baseline, and writes results/cpu.csv.

#include <chrono>
#include <cstdio>
#include <functional>
#include <string>

#include "matmul_cpu.h"
#include "matrix.h"

using MatmulFn = std::function<void(const float*, const float*, float*, int, int, int)>;

static double time_ms(const MatmulFn& fn, const Matrix& A, const Matrix& B, Matrix& C, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn(A.data.data(), B.data.data(), C.data.data(), A.rows, B.cols, A.cols);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;  // best-of-N hides one-off noise from the OS
    }
    return best;
}

int main() {
    const int sizes[] = {256, 512, 1024};
    const struct { std::string name; MatmulFn fn; } impls[] = {
        {"cpu_naive", matmul_cpu_naive},
        {"cpu_reordered", matmul_cpu_reordered},
    };

    FILE* csv = std::fopen("results/cpu.csv", "w");
    if (!csv) {
        std::fprintf(stderr, "Could not open results/cpu.csv (run from the project folder)\n");
        return 1;
    }
    std::fprintf(csv, "impl,size,ms,gflops,max_err\n");
    std::printf("%-16s %6s %12s %10s %10s  %s\n", "impl", "size", "time (ms)", "GFLOPS", "max err", "check");

    for (int n : sizes) {
        Matrix A(n, n), B(n, n), ref(n, n);
        fill_random(A, 1);
        fill_random(B, 2);
        matmul_cpu_naive(A.data.data(), B.data.data(), ref.data.data(), n, n, n);

        int reps = n <= 512 ? 3 : 1;
        for (const auto& impl : impls) {
            Matrix C(n, n);
            double ms = time_ms(impl.fn, A, B, C, reps);
            float err = max_abs_diff(C, ref);
            // A wrong answer's speed means nothing, so only show GFLOPS for passing runs.
            bool pass = err < 1e-2f;
            const char* status = pass ? "PASS" : "FAIL (not implemented yet?)";
            char speed[32] = "--";
            if (pass) std::snprintf(speed, sizeof(speed), "%.1f", gflops(n, n, n, ms));
            std::printf("%-16s %6d %12.2f %10s %10.2e  %s\n", impl.name.c_str(), n, ms, speed, err, status);
            std::fprintf(csv, "%s,%d,%.4f,%.4f,%.3e\n", impl.name.c_str(), n, ms, gflops(n, n, n, ms), err);
        }
    }
    std::fclose(csv);
    std::printf("\nWrote results/cpu.csv\n");
    return 0;
}
