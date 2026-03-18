#include <cstddef>

extern "C" __attribute__((noinline, used)) double kernel_sum_squares(const double* x, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double v = x[i];
        s += v * v;
    }
    return s;
}

