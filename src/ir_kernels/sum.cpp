#include <cstddef>

extern "C" __attribute__((noinline, used)) double kernel_sum(const double* x, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += x[i];
    }
    return s;
}

