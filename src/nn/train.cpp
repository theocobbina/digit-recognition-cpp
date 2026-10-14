// Day 2: train a 2-layer network on MNIST.
//
// Network:  input (784) -> Linear (784 x 128) -> ReLU -> Linear (128 x 10) -> softmax
// Loss:     cross-entropy
// Training: mini-batch SGD, batch size 64, learning rate ~0.1, a few epochs.
//
// Data loading is done for you. The network itself is yours to write.

#include <cstdio>

#include "mnist.h"

int main() {
    Matrix X_train = mnist::load_images("data/train-images-idx3-ubyte");
    std::vector<int> y_train = mnist::load_labels("data/train-labels-idx1-ubyte");
    Matrix X_test = mnist::load_images("data/t10k-images-idx3-ubyte");
    std::vector<int> y_test = mnist::load_labels("data/t10k-labels-idx1-ubyte");

    std::printf("Train: %d images x %d pixels\n", X_train.rows, X_train.cols);
    std::printf("Test:  %d images x %d pixels\n", X_test.rows, X_test.cols);
    std::printf("First training label: %d\n", y_train[0]);

    // TODO(Day 2), in this order. Test each step before moving on.
    //
    //  1. Parameters: W1 (784 x 128), b1 (128), W2 (128 x 10), b2 (10).
    //     Initialize W with small random values (e.g. uniform in +/- sqrt(6 / fan_in)), b with zeros.
    //
    //  2. Forward pass for one batch X (64 x 784):
    //       Z1 = X * W1 + b1        <- your matmul from Day 1
    //       H  = relu(Z1)
    //       Z2 = H * W2 + b2
    //       P  = softmax(Z2)        (subtract each row's max before exp() to avoid overflow)
    //       loss = mean of -log(P[i][label_i])
    //
    //  3. Backward pass (gradients of the loss):
    //       dZ2 = (P - one_hot(labels)) / batch_size
    //       dW2 = H^T * dZ2,    db2 = column sums of dZ2
    //       dH  = dZ2 * W2^T
    //       dZ1 = dH where Z1 > 0, else 0
    //       dW1 = X^T * dZ1,    db1 = column sums of dZ1
    //
    //  4. Update: W -= lr * dW, b -= lr * db.
    //
    //  5. After each epoch, print training loss and test accuracy (argmax of P vs label).
    //
    //  6. Then swap the big matmuls for your GPU kernel and time an epoch on CPU vs GPU.
    //
    // Sanity checks: the loss should start near ln(10) = 2.30 and fall. If it rises or
    // becomes NaN, lower the learning rate or check the softmax.

    return 0;
}
