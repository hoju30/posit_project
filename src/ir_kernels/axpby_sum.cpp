#include <cstddef>

extern "C" __attribute__((noinline, used)) double kernel_axpby_sum(double alpha, const double* x, double beta, const double* y, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += alpha * x[i] + beta * y[i];
    }
    return s;
}

