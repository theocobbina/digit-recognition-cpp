# KernelNet

A neural network library built from scratch in C++/CUDA, with hand-optimized GPU kernels benchmarked against cuBLAS and PyTorch.

> Status: in progress. CPU and MNIST results are in; GPU results are next.

## Results

| Implementation | 1024 x 1024 time (ms) | GFLOPS | Speedup vs CPU naive |
| --- | --- | --- | --- |
| CPU naive | 1,116 | 1.9 | 1x |
| CPU loop-reordered | 60.9 | 35.3 | 18x |
| GPU naive | | | |
| GPU tiled (shared memory) | | | |
| cuBLAS (reference) | | | |

![Matrix multiply throughput](results/matmul_gflops.png)

**MNIST:** 2-layer network (784 → 128 → 10) trained from scratch for 5 epochs reaches **96.99% test accuracy** on the 10,000 held-out test images (about 0.8 s per epoch on an Apple M5 CPU). PyTorch equivalent: ___%. Per-epoch numbers: [results/train.csv](results/train.csv).

Hardware: CPU results on an Apple M5 (MacBook), GPU results on an NVIDIA T4 (Google Colab). Every result is checked against a trusted reference before it is counted.

## How it works

_Write this in your own words once each part works. A few sentences each:_

- **Why the naive CPU loop is slow** (cache lines, stride)
- **How the GPU naive kernel maps threads to outputs**
- **Why tiling with shared memory is faster** (global-memory reads cut by about TILE times)
- **How backpropagation works in the 2-layer network**

## Project layout

```
include/          headers: Matrix type, kernel declarations, MNIST loader, CUDA error checks
src/cpu/          CPU matrix multiply (naive + reordered)
src/gpu/          CUDA kernels (naive + tiled) and the cuBLAS reference
src/nn/           MNIST training program (Day 2)
bench/            benchmark programs that time and verify every implementation
scripts/          MNIST download and the benchmark plot
notebooks/        Google Colab notebook for running the GPU code
results/          benchmark CSVs and charts
```

## Running it

**CPU (on your Mac or Linux):**

```bash
make run-cpu
```

**GPU (Google Colab, free T4 GPU):** open `notebooks/kernelnet_colab.ipynb` in Colab, switch the runtime to T4 GPU, and run the cells.

**Training (Day 2):**

```bash
python3 scripts/download_mnist.py
make run-train
```

Each benchmark checks its answer against a trusted reference (naive CPU or cuBLAS) and prints PASS or FAIL, so a fast but wrong kernel never counts.

## Roadmap

**Day 1: matrix multiply**
- [x] `matmul_cpu_reordered` in `src/cpu/matmul_cpu.cpp` passes and beats naive
- [ ] `matmul_naive_kernel` in `src/gpu/matmul_gpu.cu` passes
- [ ] `matmul_tiled_kernel` passes and beats the naive kernel
- [ ] Benchmark chart generated and results table filled in

**Day 2: neural network**
- [x] Forward pass: linear, ReLU, softmax, cross-entropy loss
- [x] Backward pass and SGD update; loss falls from about 2.30
- [x] ~95%+ test accuracy on MNIST
- [ ] Same model in PyTorch for comparison
- [ ] "How it works" section written

**Next steps**
- [ ] Larger tiles and register blocking (each thread computes several outputs)
- [ ] Run the whole forward/backward pass on the GPU
- [ ] Profile with Nsight Compute
