#include <math.h>
#include <stddef.h>
#include <stdint.h>

static volatile double sink_stencil1d_like_v12 = 0.0;

static inline double clamp_small(double x) {
  return (fabs(x) < 1e-12) ? 0.0 : x;
}

#define N 112
#define STEPS 4

int main(void) {
  double a[N], b[N];
  for (int i = 0; i < N; ++i) a[i] = sin(0.06 * (double)(i + 1));
  for (int t = 0; t < STEPS; ++t) {
    for (int i = 1; i < N - 1; ++i)
      b[i] = 0.25 * a[i - 1] + 0.5 * a[i] + 0.25 * a[i + 1];
    for (int i = 1; i < N - 1; ++i)
      a[i] = b[i];
  }
  double acc = 0.0;
  for (int i = 1; i < N - 1; ++i) acc += a[i];
  sink_stencil1d_like_v12 = clamp_small(acc);
  return (int)fabs(acc);
}
