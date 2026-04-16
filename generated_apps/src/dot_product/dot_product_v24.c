#include <math.h>
#include <stddef.h>
#include <stdint.h>

static volatile double sink_dot_product_v24 = 0.0;

static inline double clamp_small(double x) {
  return (fabs(x) < 1e-12) ? 0.0 : x;
}

#define N 288

int main(void) {
  double x[N], y[N];
  for (int i = 0; i < N; ++i) {
    x[i] = cos((0.21) * (double)(i + 1));
    y[i] = sin((0.44) * (double)(i + 3));
  }
  double acc = 0.0;
  for (int i = 0; i < N; ++i)
    acc += x[i] * y[i];
  sink_dot_product_v24 = clamp_small(acc);
  return (int)fabs(acc);
}
