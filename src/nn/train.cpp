// Train a 2-layer network on MNIST, with no ML framework.
//
// Network:  input (784) -> Linear (784 x 128) -> ReLU -> Linear (128 x 10) -> softmax
// Loss:     cross-entropy
// Training: mini-batch SGD, batch size 64, learning rate 0.1, 5 epochs.
//
// Every matrix multiply goes through matmul_cpu_reordered from src/cpu/matmul_cpu.cpp.
// Per-epoch results are written to results/train.csv.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "matmul_cpu.h"
#include "mnist.h"

constexpr int kInput = 784;
constexpr int kHidden = 128;
constexpr int kClasses = 10;
constexpr int kBatch = 64;
constexpr int kEpochs = 5;
constexpr float kLearningRate = 0.1f;

// C = A * B
static void matmul(const Matrix& A, const Matrix& B, Matrix& C) {
    matmul_cpu_reordered(A.data.data(), B.data.data(), C.data.data(), A.rows, B.cols, A.cols);
}

static Matrix transpose(const Matrix& m) {
    Matrix t(m.cols, m.rows);
    for (int r = 0; r < m.rows; ++r)
        for (int c = 0; c < m.cols; ++c) t.at(c, r) = m.at(r, c);
    return t;
}

// Uniform in +/- sqrt(6 / fan_in) (He initialization), which suits ReLU layers.
static void init_weights(Matrix& W, std::mt19937& rng) {
    float limit = std::sqrt(6.0f / W.rows);
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (float& w : W.data) w = dist(rng);
}

static void add_bias(Matrix& Z, const std::vector<float>& b) {
    for (int r = 0; r < Z.rows; ++r)
        for (int c = 0; c < Z.cols; ++c) Z.at(r, c) += b[c];
}

// Softmax each row in place. Subtracting the row max first keeps exp() from overflowing.
static void softmax_rows(Matrix& Z) {
    for (int r = 0; r < Z.rows; ++r) {
        float mx = Z.at(r, 0);
        for (int c = 1; c < Z.cols; ++c) mx = std::max(mx, Z.at(r, c));
        float sum = 0.0f;
        for (int c = 0; c < Z.cols; ++c) {
            Z.at(r, c) = std::exp(Z.at(r, c) - mx);
            sum += Z.at(r, c);
        }
        for (int c = 0; c < Z.cols; ++c) Z.at(r, c) /= sum;
    }
}

struct Network {
    Matrix W1{kInput, kHidden}, W2{kHidden, kClasses};
    std::vector<float> b1 = std::vector<float>(kHidden, 0.0f);
    std::vector<float> b2 = std::vector<float>(kClasses, 0.0f);
};

// Forward pass. Z1 (pre-ReLU) and H (post-ReLU) are kept for the backward pass.
static Matrix forward(const Network& net, const Matrix& X, Matrix& Z1, Matrix& H) {
    Z1 = Matrix(X.rows, kHidden);
    matmul(X, net.W1, Z1);
    add_bias(Z1, net.b1);

    H = Z1;
    for (float& h : H.data) h = std::max(h, 0.0f);

    Matrix P(X.rows, kClasses);
    matmul(H, net.W2, P);
    add_bias(P, net.b2);
    softmax_rows(P);
    return P;
}

static float accuracy(const Network& net, const Matrix& X, const std::vector<int>& y) {
    Matrix Z1, H;
    Matrix P = forward(net, X, Z1, H);
    int correct = 0;
    for (int r = 0; r < P.rows; ++r) {
        const float* row = &P.data[static_cast<size_t>(r) * kClasses];
        int pred = static_cast<int>(std::max_element(row, row + kClasses) - row);
        correct += (pred == y[r]);
    }
    return 100.0f * correct / P.rows;
}

// One SGD step on a batch. Returns the batch's mean cross-entropy loss.
static float train_step(Network& net, const Matrix& X, const std::vector<int>& y) {
    const int n = X.rows;
    Matrix Z1, H;
    Matrix P = forward(net, X, Z1, H);

    float loss = 0.0f;
    for (int r = 0; r < n; ++r) loss -= std::log(std::max(P.at(r, y[r]), 1e-12f));
    loss /= n;

    // Softmax + cross-entropy gradient: dZ2 = (P - one_hot(y)) / n
    Matrix dZ2 = P;
    for (int r = 0; r < n; ++r) dZ2.at(r, y[r]) -= 1.0f;
    for (float& v : dZ2.data) v /= n;

    // Layer 2: dW2 = H^T * dZ2, db2 = column sums of dZ2, dH = dZ2 * W2^T
    Matrix dW2(kHidden, kClasses);
    matmul(transpose(H), dZ2, dW2);
    std::vector<float> db2(kClasses, 0.0f);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < kClasses; ++c) db2[c] += dZ2.at(r, c);
    Matrix dH(n, kHidden);
    matmul(dZ2, transpose(net.W2), dH);

    // ReLU passes gradient only where its input was positive.
    for (size_t i = 0; i < dH.size(); ++i)
        if (Z1.data[i] <= 0.0f) dH.data[i] = 0.0f;

    // Layer 1: dW1 = X^T * dZ1, db1 = column sums of dZ1
    Matrix dW1(kInput, kHidden);
    matmul(transpose(X), dH, dW1);
    std::vector<float> db1(kHidden, 0.0f);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < kHidden; ++c) db1[c] += dH.at(r, c);

    for (size_t i = 0; i < net.W1.size(); ++i) net.W1.data[i] -= kLearningRate * dW1.data[i];
    for (size_t i = 0; i < net.W2.size(); ++i) net.W2.data[i] -= kLearningRate * dW2.data[i];
    for (int c = 0; c < kHidden; ++c) net.b1[c] -= kLearningRate * db1[c];
    for (int c = 0; c < kClasses; ++c) net.b2[c] -= kLearningRate * db2[c];
    return loss;
}

int main() {
    Matrix X_train = mnist::load_images("data/train-images-idx3-ubyte");
    std::vector<int> y_train = mnist::load_labels("data/train-labels-idx1-ubyte");
    Matrix X_test = mnist::load_images("data/t10k-images-idx3-ubyte");
    std::vector<int> y_test = mnist::load_labels("data/t10k-labels-idx1-ubyte");
    std::printf("Train: %d images, Test: %d images\n\n", X_train.rows, X_test.rows);

    std::mt19937 rng(42);  // fixed seed: anyone who reruns this gets the same numbers
    Network net;
    init_weights(net.W1, rng);
    init_weights(net.W2, rng);

    FILE* csv = std::fopen("results/train.csv", "w");
    if (!csv) {
        std::fprintf(stderr, "Could not open results/train.csv (run from the kernelnet folder)\n");
        return 1;
    }
    std::fprintf(csv, "epoch,loss,train_acc,test_acc,seconds\n");
    std::printf("%5s %8s %10s %10s %8s\n", "epoch", "loss", "train acc", "test acc", "time(s)");

    std::vector<int> order(X_train.rows);
    std::iota(order.begin(), order.end(), 0);
    Matrix Xb(kBatch, kInput);
    std::vector<int> yb(kBatch);

    for (int epoch = 1; epoch <= kEpochs; ++epoch) {
        auto t0 = std::chrono::steady_clock::now();
        std::shuffle(order.begin(), order.end(), rng);

        double loss_sum = 0.0;
        int batches = X_train.rows / kBatch;  // drop the last partial batch
        for (int b = 0; b < batches; ++b) {
            for (int i = 0; i < kBatch; ++i) {
                int src = order[b * kBatch + i];
                std::copy_n(&X_train.data[static_cast<size_t>(src) * kInput], kInput,
                            &Xb.data[static_cast<size_t>(i) * kInput]);
                yb[i] = y_train[src];
            }
            loss_sum += train_step(net, Xb, yb);
        }
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        float train_acc = accuracy(net, X_train, y_train);
        float test_acc = accuracy(net, X_test, y_test);
        double loss = loss_sum / batches;
        std::printf("%5d %8.4f %9.2f%% %9.2f%% %8.1f\n", epoch, loss, train_acc, test_acc, secs);
        std::fprintf(csv, "%d,%.4f,%.2f,%.2f,%.2f\n", epoch, loss, train_acc, test_acc, secs);
    }
    std::fclose(csv);
    std::printf("\nWrote results/train.csv\n");
    return 0;
}
