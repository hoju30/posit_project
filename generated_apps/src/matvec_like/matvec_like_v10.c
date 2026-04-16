#include <math.h>
#include <stddef.h>
#include <stdint.h>

static volatile double sink_matvec_like_v10 = 0.0;

static inline double clamp_small(double x) {
  return (fabs(x) < 1e-12) ? 0.0 : x;
}

#define ROWS 40
#define COLS 40

int main(void) {
  double A[ROWS][COLS];
  double x[COLS];
  double y[ROWS];
  for (int i = 0; i < ROWS; ++i)
    for (int j = 0; j < COLS; ++j)
      A[i][j] = ((double)((i + 1) * (j + 2))) / (double)(ROWS + COLS);
  for (int j = 0; j < COLS; ++j)
    x[j] = sin(0.03 * (double)(j + 1));
  for (int i = 0; i < ROWS; ++i) {
    double acc = 0.0;
    for (int j = 0; j < COLS; ++j)
      acc += A[i][j] * x[j];
    y[i] = acc;
  }
  double total = 0.0;
  for (int i = 0; i < ROWS; ++i) total += y[i];
  sink_matvec_like_v10 = clamp_small(total);
  return (int)fabs(total);
}
