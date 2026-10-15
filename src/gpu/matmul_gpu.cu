#include "cuda_utils.cuh"
#include "matmul_gpu.cuh"

// ---------------------------------------------------------------------------
// Kernel 1: naive. One thread computes one element C[row][col].
// ---------------------------------------------------------------------------
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Interview question: every thread reads a full row of A and a full column of B
    // from global memory. How many times is each element of A read in total? (N times)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;  // the grid is rounded up past the matrix edge

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

void launch_matmul_naive(const float* dA, const float* dB, float* dC, int M, int N, int K) {
    dim3 block(16, 16);  // 256 threads per block
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel 2: tiled. Each block loads TILE x TILE pieces of A and B into fast
// shared memory, so each value fetched from slow global memory is reused TILE times.
// ---------------------------------------------------------------------------
constexpr int TILE = 16;

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Interview questions: why are both __syncthreads() calls needed?
    // Why does this cut global-memory traffic by about TILE times?
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // Each thread loads one element of each tile; zeros pad past the matrix edge.
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();  // whole tile loaded before anyone reads it

        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();  // everyone done reading before the next tile overwrites it
    }

    if (row < M && col < N) C[row * N + col] = sum;
}

void launch_matmul_tiled(const float* dA, const float* dB, float* dC, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Reference: cuBLAS.
// ---------------------------------------------------------------------------
void launch_matmul_cublas(cublasHandle_t handle, const float* dA, const float* dB, float* dC, int M, int N, int K) {
    // cuBLAS is column-major. A row-major matrix read as column-major is its transpose,
    // so we ask for C^T = B^T * A^T, which leaves C in row-major order.
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha, dB, N, dA, K, &beta, dC, N));
}
