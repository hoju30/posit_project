#include <math.h>
#include <stddef.h>
#include <stdint.h>

static volatile double sink_two_stage_reduce_v15 = 0.0;

static inline double clamp_small(double x) {
  return (fabs(x) < 1e-12) ? 0.0 : x;
}

#define N 208

int main(void) {
  double x[N], tmp[N];
  for (int i = 0; i < N; ++i)
    x[i] = sin(0.04 * (double)(i + 1)) + cos(0.02 * (double)(i + 1));
  for (int i = 0; i < N; ++i)
    tmp[i] = x[i] * x[i] + 0.125 * x[i];
  double acc = 0.0;
  for (int i = 0; i < N; ++i)
    acc += tmp[i];
  sink_two_stage_reduce_v15 = clamp_small(acc);
  return (int)fabs(acc);
}
