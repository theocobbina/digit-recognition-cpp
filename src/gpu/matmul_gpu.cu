#include "cuda_utils.cuh"
#include "matmul_gpu.cuh"

// ---------------------------------------------------------------------------
// Kernel 1: naive. One thread computes one element C[row][col].
// ---------------------------------------------------------------------------
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO(Day 1):
    //  1. Work out which element this thread owns:
    //       row = blockIdx.y * blockDim.y + threadIdx.y
    //       col = blockIdx.x * blockDim.x + threadIdx.x
    //  2. The grid is rounded up, so some threads fall outside the matrix.
    //     Return early if row >= M or col >= N.
    //  3. Loop k from 0 to K, accumulate A[row*K + k] * B[k*N + col], write C[row*N + col].
    //
    // Interview question: every thread reads a full row of A and a full column of B
    // from global memory. How many times is each element of A read in total?
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
    // TODO(Day 1):
    //  1. Declare the shared tiles:
    //       __shared__ float As[TILE][TILE];
    //       __shared__ float Bs[TILE][TILE];
    //  2. Compute row/col as in the naive kernel (block size is TILE x TILE).
    //  3. Loop over tiles t = 0 .. ceil(K / TILE):
    //       a. Each thread loads ONE element of A's tile and ONE of B's tile:
    //            As[ty][tx] = A[row * K + (t*TILE + tx)]   (0.0f if out of bounds)
    //            Bs[ty][tx] = B[(t*TILE + ty) * N + col]   (0.0f if out of bounds)
    //       b. __syncthreads();   // wait until the whole tile is loaded
    //       c. for k in 0..TILE: sum += As[ty][k] * Bs[k][tx];
    //       d. __syncthreads();   // wait before the next loop overwrites the tiles
    //  4. If row < M and col < N, write C[row * N + col] = sum.
    //
    // Interview questions: why are both __syncthreads() calls needed?
    // Why does this cut global-memory traffic by about TILE times?
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
