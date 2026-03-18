#include <cstddef>

extern "C" __attribute__((noinline, used)) double kernel_prefix_sum_total(const double* x, std::size_t n) {
    double running = 0.0;
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        running += x[i];
        total += running;
    }
    return total;
}

