# Build.
#   make cpu        build the matrix-multiply benchmark
#   make train      build the MNIST trainer
#   make run-cpu / run-train / plot / clean

CXX       ?= c++
CXXFLAGS  := -O2 -std=c++17 -Wall -Wextra -Iinclude

CPU_SRC := src/cpu/matmul_cpu.cpp

.PHONY: cpu train run-cpu run-train plot clean

cpu: build/bench_cpu
train: build/train

build results:
	mkdir -p $@

build/bench_cpu: bench/bench_cpu.cpp $(CPU_SRC) $(wildcard include/*.h) | build
	$(CXX) $(CXXFLAGS) bench/bench_cpu.cpp $(CPU_SRC) -o $@

build/train: src/nn/train.cpp $(CPU_SRC) $(wildcard include/*.h) | build
	$(CXX) $(CXXFLAGS) src/nn/train.cpp $(CPU_SRC) -o $@

run-cpu: cpu | results
	./build/bench_cpu

run-train: train
	./build/train

plot:
	./.venv/bin/python scripts/plot_bench.py
	./.venv/bin/python scripts/plot_compare.py

clean:
	rm -rf build
