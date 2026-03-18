#include <cstddef>

extern "C" __attribute__((noinline, used)) double kernel_l1_norm(const double* x, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double v = x[i];
        s += (v >= 0.0) ? v : -v;
    }
    return s;
}

