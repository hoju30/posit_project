#include <math.h>
#include <stddef.h>
#include <stdint.h>

static volatile double sink_weighted_sum_v10 = 0.0;

static inline double clamp_small(double x) {
  return (fabs(x) < 1e-12) ? 0.0 : x;
}

#define N 208

int main(void) {
  double x[N], y[N];
  for (int i = 0; i < N; ++i) {
    x[i] = sin(0.07 * (double)(i + 1));
    y[i] = cos(0.11 * (double)(i + 2));
  }
  double acc = 0.0;
  for (int i = 0; i < N; ++i)
    acc += (0.5) * x[i] + (-0.04999999999999999) * y[i];
  sink_weighted_sum_v10 = clamp_small(acc);
  return (int)fabs(acc);
}
