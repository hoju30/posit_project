#include <math.h>
#include <stddef.h>
#include <stdint.h>

static volatile double sink_ema_recurrence_v21 = 0.0;

static inline double clamp_small(double x) {
  return (fabs(x) < 1e-12) ? 0.0 : x;
}

#define N 120

int main(void) {
  double x[N];
  for (int i = 0; i < N; ++i)
    x[i] = cos(0.09 * (double)(i + 1)) + 0.01 * (double)(i % 5);
  double ema = x[0];
  for (int i = 1; i < N; ++i)
    ema = (0.14) * x[i] + (1.0 - (0.14)) * ema;
  sink_ema_recurrence_v21 = clamp_small(ema);
  return (int)fabs(ema);
}
