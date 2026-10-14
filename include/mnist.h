#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "matrix.h"

// Loads MNIST from the raw IDX files that scripts/download_mnist.py saves in data/.
// Images become a (count x 784) Matrix with pixels scaled to [0, 1].
// Labels become a vector of digits 0-9.

namespace mnist {

inline uint32_t read_be32(FILE* f) {
    unsigned char b[4];
    if (std::fread(b, 1, 4, f) != 4) {
        std::fprintf(stderr, "MNIST: unexpected end of file\n");
        std::exit(1);
    }
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

inline FILE* open_or_die(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "Could not open %s. Run: python3 scripts/download_mnist.py\n", path.c_str());
        std::exit(1);
    }
    return f;
}

inline Matrix load_images(const std::string& path) {
    FILE* f = open_or_die(path);
    if (read_be32(f) != 2051) { std::fprintf(stderr, "%s is not an MNIST image file\n", path.c_str()); std::exit(1); }
    int count = static_cast<int>(read_be32(f));
    int rows = static_cast<int>(read_be32(f));
    int cols = static_cast<int>(read_be32(f));

    std::vector<unsigned char> pixels(static_cast<size_t>(count) * rows * cols);
    if (std::fread(pixels.data(), 1, pixels.size(), f) != pixels.size()) { std::fprintf(stderr, "MNIST: truncated %s\n", path.c_str()); std::exit(1); }
    std::fclose(f);

    Matrix X(count, rows * cols);
    for (size_t i = 0; i < pixels.size(); ++i) X.data[i] = pixels[i] / 255.0f;
    return X;
}

inline std::vector<int> load_labels(const std::string& path) {
    FILE* f = open_or_die(path);
    if (read_be32(f) != 2049) { std::fprintf(stderr, "%s is not an MNIST label file\n", path.c_str()); std::exit(1); }
    int count = static_cast<int>(read_be32(f));

    std::vector<unsigned char> raw(count);
    if (std::fread(raw.data(), 1, raw.size(), f) != raw.size()) { std::fprintf(stderr, "MNIST: truncated %s\n", path.c_str()); std::exit(1); }
    std::fclose(f);
    return std::vector<int>(raw.begin(), raw.end());
}

}  // namespace mnist
