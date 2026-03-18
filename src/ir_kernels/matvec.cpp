#include <cstddef>

// A is row-major, shape = (rows, cols)
extern "C" __attribute__((noinline, used)) double kernel_matvec_sum(const double* A, const double* x, std::size_t rows, std::size_t cols) {
    double total = 0.0;
    for (std::size_t r = 0; r < rows; ++r) {
        double s = 0.0;
        const std::size_t base = r * cols;
        for (std::size_t c = 0; c < cols; ++c) {
            s += A[base + c] * x[c];
        }
        total += s;
    }
    return total;
}

