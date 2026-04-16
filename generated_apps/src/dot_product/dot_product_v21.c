#include <math.h>
#include <stddef.h>
#include <stdint.h>

static volatile double sink_dot_product_v21 = 0.0;

static inline double clamp_small(double x) {
  return (fabs(x) < 1e-12) ? 0.0 : x;
}

#define N 192

int main(void) {
  double x[N], y[N];
  for (int i = 0; i < N; ++i) {
    x[i] = cos((0.44999999999999996) * (double)(i + 1));
    y[i] = sin((0.32) * (double)(i + 3));
  }
  double acc = 0.0;
  for (int i = 0; i < N; ++i)
    acc += x[i] * y[i];
  sink_dot_product_v21 = clamp_small(acc);
  return (int)fabs(acc);
}
