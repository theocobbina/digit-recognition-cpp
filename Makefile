# KernelNet build.
#   make cpu        build the CPU benchmark (works on Mac/Linux)
#   make gpu        build the GPU benchmark (needs nvcc: use Google Colab)
#   make train      build the MNIST trainer (Day 2)
#   make run-cpu / run-gpu / run-train / plot / clean

CXX       ?= c++
CXXFLAGS  := -O2 -std=c++17 -Wall -Wextra -Iinclude
NVCC      ?= nvcc
# sm_75 = Turing (the T4 GPU on free Colab). Change it if you use a different GPU.
NVCCFLAGS := -O2 -std=c++17 -Iinclude -arch=sm_75

CPU_SRC := src/cpu/matmul_cpu.cpp
GPU_SRC := src/gpu/matmul_gpu.cu

.PHONY: cpu gpu train run-cpu run-gpu run-train plot clean

cpu: build/bench_cpu
gpu: build/bench_gpu
train: build/train

build results:
	mkdir -p $@

build/bench_cpu: bench/bench_cpu.cpp $(CPU_SRC) $(wildcard include/*.h) | build
	$(CXX) $(CXXFLAGS) bench/bench_cpu.cpp $(CPU_SRC) -o $@

build/bench_gpu: bench/bench_gpu.cu $(GPU_SRC) $(wildcard include/*.h include/*.cuh) | build
	$(NVCC) $(NVCCFLAGS) bench/bench_gpu.cu $(GPU_SRC) -lcublas -o $@

build/train: src/nn/train.cpp $(CPU_SRC) $(wildcard include/*.h) | build
	$(CXX) $(CXXFLAGS) src/nn/train.cpp $(CPU_SRC) -o $@

run-cpu: cpu | results
	./build/bench_cpu

run-gpu: gpu | results
	./build/bench_gpu

run-train: train
	./build/train

plot:
	python3 scripts/plot_bench.py

clean:
	rm -rf build
