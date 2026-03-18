#include <cstddef>

extern "C" __attribute__((noinline, used)) double kernel_axpy_sum(double alpha, const double* x, const double* y, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += alpha * x[i] + y[i];
    }
    return s;
}

