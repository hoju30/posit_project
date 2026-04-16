#include <math.h>
#include <stddef.h>
#include <stdint.h>

static volatile double sink_prefix_recurrence_v8 = 0.0;

static inline double clamp_small(double x) {
  return (fabs(x) < 1e-12) ? 0.0 : x;
}

#define N 224

int main(void) {
  double x[N], out[N];
  for (int i = 0; i < N; ++i)
    x[i] = sin(0.05 * (double)(i + 1));
  double state = 0.0;
  for (int i = 0; i < N; ++i) {
    state = (0.85) * state + x[i];
    out[i] = state;
  }
  sink_prefix_recurrence_v8 = clamp_small(out[N - 1]);
  return (int)fabs(out[N - 1]);
}
