#include <math.h>
#include <stddef.h>
#include <stdint.h>

static volatile double sink_sum_reduce_v11 = 0.0;

static inline double clamp_small(double x) {
  return (fabs(x) < 1e-12) ? 0.0 : x;
}

#define N 304

int main(void) {
  double x[N];
  for (int i = 0; i < N; ++i)
    x[i] = sin((0.45) * (double)(i + 1)) + (-0.03125);
  double acc = 0.0;
  for (int i = 0; i < N; ++i)
    acc += x[i];
  sink_sum_reduce_v11 = clamp_small(acc);
  return (int)fabs(acc);
}
