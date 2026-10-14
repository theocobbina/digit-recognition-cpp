// GPU matrix-multiply benchmark.
// Times each kernel with CUDA events, checks it against cuBLAS, and writes results/gpu.csv.

#include <cstdio>
#include <string>

#include "cuda_utils.cuh"
#include "matmul_gpu.cuh"
#include "matrix.h"

enum class Impl { Naive, Tiled, Cublas };

static void run(Impl impl, cublasHandle_t h, const float* dA, const float* dB, float* dC, int n) {
    switch (impl) {
        case Impl::Naive:  launch_matmul_naive(dA, dB, dC, n, n, n); break;
        case Impl::Tiled:  launch_matmul_tiled(dA, dB, dC, n, n, n); break;
        case Impl::Cublas: launch_matmul_cublas(h, dA, dB, dC, n, n, n); break;
    }
}

int main() {
    const int sizes[] = {512, 1024, 2048};
    const struct { std::string name; Impl impl; } impls[] = {
        {"gpu_naive", Impl::Naive},
        {"gpu_tiled", Impl::Tiled},
        {"cublas", Impl::Cublas},
    };
    const int reps = 10;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s\n\n", prop.name);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    FILE* csv = std::fopen("results/gpu.csv", "w");
    if (!csv) {
        std::fprintf(stderr, "Could not open results/gpu.csv (run from the kernelnet folder)\n");
        return 1;
    }
    std::fprintf(csv, "impl,size,ms,gflops,max_err\n");
    std::printf("%-12s %6s %12s %10s %10s  %s\n", "impl", "size", "time (ms)", "GFLOPS", "max err", "check");

    for (int n : sizes) {
        Matrix A(n, n), B(n, n), ref(n, n), C(n, n);
        fill_random(A, 1);
        fill_random(B, 2);
        size_t bytes = A.size() * sizeof(float);

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));
        CUDA_CHECK(cudaMemcpy(dA, A.data.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data.data(), bytes, cudaMemcpyHostToDevice));

        // cuBLAS result is the trusted answer for this size.
        launch_matmul_cublas(handle, dA, dB, dC, n, n, n);
        CUDA_CHECK(cudaMemcpy(ref.data.data(), dC, bytes, cudaMemcpyDeviceToHost));

        for (const auto& impl : impls) {
            CUDA_CHECK(cudaMemset(dC, 0, bytes));
            run(impl.impl, handle, dA, dB, dC, n);  // warm-up, also the run we check
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(C.data.data(), dC, bytes, cudaMemcpyDeviceToHost));
            float err = max_abs_diff(C, ref);

            // Kernel launches return immediately, so time on the GPU with events.
            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < reps; ++r) run(impl.impl, handle, dA, dB, dC, n);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float total_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
            double ms = total_ms / reps;

            // A wrong answer's speed means nothing, so only show GFLOPS for passing runs.
            bool pass = err < 1e-2f;
            const char* status = pass ? "PASS" : "FAIL (not implemented yet?)";
            char speed[32] = "--";
            if (pass) std::snprintf(speed, sizeof(speed), "%.1f", gflops(n, n, n, ms));
            std::printf("%-12s %6d %12.3f %10s %10.2e  %s\n", impl.name.c_str(), n, ms, speed, err, status);
            std::fprintf(csv, "%s,%d,%.4f,%.4f,%.3e\n", impl.name.c_str(), n, ms, gflops(n, n, n, ms), err);
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    std::fclose(csv);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    std::printf("\nWrote results/gpu.csv\n");
    return 0;
}
