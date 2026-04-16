#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FAMILIES = [
    "sum_reduce",
    "dot_product",
    "weighted_sum",
    "prefix_recurrence",
    "ema_recurrence",
    "matvec_like",
    "two_stage_reduce",
    "stencil1d_like",
]


def common_prelude(name: str) -> str:
    return f"""#include <math.h>
#include <stddef.h>
#include <stdint.h>

static volatile double sink_{name} = 0.0;

static inline double clamp_small(double x) {{
  return (fabs(x) < 1e-12) ? 0.0 : x;
}}
"""


def emit_sum_reduce(family: str, vid: int) -> str:
    n = 128 + 16 * (vid % 13)
    scale = 0.25 + 0.05 * (vid % 7)
    bias = ((vid % 5) - 2) * 0.03125
    return common_prelude(f"{family}_v{vid}") + f"""
#define N {n}

int main(void) {{
  double x[N];
  for (int i = 0; i < N; ++i)
    x[i] = sin(({scale}) * (double)(i + 1)) + ({bias});
  double acc = 0.0;
  for (int i = 0; i < N; ++i)
    acc += x[i];
  sink_{family}_v{vid} = clamp_small(acc);
  return (int)fabs(acc);
}}
"""


def emit_dot_product(family: str, vid: int) -> str:
    n = 96 + 32 * (vid % 9)
    a = 0.15 + 0.03 * (vid % 11)
    b = 0.2 + 0.04 * ((vid + 3) % 7)
    return common_prelude(f"{family}_v{vid}") + f"""
#define N {n}

int main(void) {{
  double x[N], y[N];
  for (int i = 0; i < N; ++i) {{
    x[i] = cos(({a}) * (double)(i + 1));
    y[i] = sin(({b}) * (double)(i + 3));
  }}
  double acc = 0.0;
  for (int i = 0; i < N; ++i)
    acc += x[i] * y[i];
  sink_{family}_v{vid} = clamp_small(acc);
  return (int)fabs(acc);
}}
"""


    n = 128 + 8 * (vid % 17)
def emit_weighted_sum(family: str, vid: int) -> str:
    alpha = 0.5 + 0.1 * (vid % 5)
    beta = -0.25 + 0.05 * ((vid + 1) % 7)
    return common_prelude(f"{family}_v{vid}") + f"""
#define N {n}

int main(void) {{
  double x[N], y[N];
  for (int i = 0; i < N; ++i) {{
    x[i] = sin(0.07 * (double)(i + 1));
    y[i] = cos(0.11 * (double)(i + 2));
  }}
  double acc = 0.0;
  for (int i = 0; i < N; ++i)
    acc += ({alpha}) * x[i] + ({beta}) * y[i];
  sink_{family}_v{vid} = clamp_small(acc);
  return (int)fabs(acc);
}}
"""


def emit_prefix_recurrence(family: str, vid: int) -> str:
    n = 96 + 16 * (vid % 12)
    gamma = 0.85 + 0.01 * (vid % 8)
    return common_prelude(f"{family}_v{vid}") + f"""
#define N {n}

int main(void) {{
  double x[N], out[N];
  for (int i = 0; i < N; ++i)
    x[i] = sin(0.05 * (double)(i + 1));
  double state = 0.0;
  for (int i = 0; i < N; ++i) {{
    state = ({gamma}) * state + x[i];
    out[i] = state;
  }}
  sink_{family}_v{vid} = clamp_small(out[N - 1]);
  return (int)fabs(out[N - 1]);
}}
"""


def emit_ema_recurrence(family: str, vid: int) -> str:
    n = 100 + 20 * (vid % 10)
    alpha = 0.05 + 0.03 * (vid % 6)
    return common_prelude(f"{family}_v{vid}") + f"""
#define N {n}

int main(void) {{
  double x[N];
  for (int i = 0; i < N; ++i)
    x[i] = cos(0.09 * (double)(i + 1)) + 0.01 * (double)(i % 5);
  double ema = x[0];
  for (int i = 1; i < N; ++i)
    ema = ({alpha}) * x[i] + (1.0 - ({alpha})) * ema;
  sink_{family}_v{vid} = clamp_small(ema);
  return (int)fabs(ema);
}}
"""


def emit_matvec_like(family: str, vid: int) -> str:
    rows = 24 + 4 * (vid % 6)
    cols = 24 + 8 * ((vid + 2) % 5)
    return common_prelude(f"{family}_v{vid}") + f"""
#define ROWS {rows}
#define COLS {cols}

int main(void) {{
  double A[ROWS][COLS];
  double x[COLS];
  double y[ROWS];
  for (int i = 0; i < ROWS; ++i)
    for (int j = 0; j < COLS; ++j)
      A[i][j] = ((double)((i + 1) * (j + 2))) / (double)(ROWS + COLS);
  for (int j = 0; j < COLS; ++j)
    x[j] = sin(0.03 * (double)(j + 1));
  for (int i = 0; i < ROWS; ++i) {{
    double acc = 0.0;
    for (int j = 0; j < COLS; ++j)
      acc += A[i][j] * x[j];
    y[i] = acc;
  }}
  double total = 0.0;
  for (int i = 0; i < ROWS; ++i) total += y[i];
  sink_{family}_v{vid} = clamp_small(total);
  return (int)fabs(total);
}}
"""


def emit_two_stage_reduce(family: str, vid: int) -> str:
    n = 128 + 16 * (vid % 10)
    return common_prelude(f"{family}_v{vid}") + f"""
#define N {n}

int main(void) {{
  double x[N], tmp[N];
  for (int i = 0; i < N; ++i)
    x[i] = sin(0.04 * (double)(i + 1)) + cos(0.02 * (double)(i + 1));
  for (int i = 0; i < N; ++i)
    tmp[i] = x[i] * x[i] + 0.125 * x[i];
  double acc = 0.0;
  for (int i = 0; i < N; ++i)
    acc += tmp[i];
  sink_{family}_v{vid} = clamp_small(acc);
  return (int)fabs(acc);
}}
"""


def emit_stencil1d_like(family: str, vid: int) -> str:
    n = 96 + 16 * (vid % 11)
    steps = 4 + (vid % 4)
    return common_prelude(f"{family}_v{vid}") + f"""
#define N {n}
#define STEPS {steps}

int main(void) {{
  double a[N], b[N];
  for (int i = 0; i < N; ++i) a[i] = sin(0.06 * (double)(i + 1));
  for (int t = 0; t < STEPS; ++t) {{
    for (int i = 1; i < N - 1; ++i)
      b[i] = 0.25 * a[i - 1] + 0.5 * a[i] + 0.25 * a[i + 1];
    for (int i = 1; i < N - 1; ++i)
      a[i] = b[i];
  }}
  double acc = 0.0;
  for (int i = 1; i < N - 1; ++i) acc += a[i];
  sink_{family}_v{vid} = clamp_small(acc);
  return (int)fabs(acc);
}}
"""


EMITTERS = {
    "sum_reduce": emit_sum_reduce,
    "dot_product": emit_dot_product,
    "weighted_sum": emit_weighted_sum,
    "prefix_recurrence": emit_prefix_recurrence,
    "ema_recurrence": emit_ema_recurrence,
    "matvec_like": emit_matvec_like,
    "two_stage_reduce": emit_two_stage_reduce,
    "stencil1d_like": emit_stencil1d_like,
}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/hoju/test/posit_test/generated_apps", help="root output dir")
    ap.add_argument("--variants-per-family", type=int, default=25, help="default 25 => 8*25 = 200 apps")
    ap.add_argument("--families", default=",".join(FAMILIES), help="comma-separated family list")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    families = [f.strip() for f in str(args.families).split(",") if f.strip()]
    bad = [f for f in families if f not in EMITTERS]
    if bad:
        raise SystemExit(f"unknown families: {bad}")

    src_root = out_dir / "src"
    manifest_rows: list[dict[str, str]] = []
    total = 0
    for family in families:
        emitter = EMITTERS[family]
        for vid in range(int(args.variants_per_family)):
            app_name = f"{family}_v{vid:02d}"
            rel = Path(family) / f"{app_name}.c"
            path = src_root / rel
            write_text(path, emitter(family, vid))
            manifest_rows.append(
                {
                    "app": app_name,
                    "family": family,
                    "variant_id": f"v{vid:02d}",
                    "source_path": str(path.resolve()),
                    "language": "c",
                }
            )
            total += 1

    manifest_path = out_dir / "apps_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["app", "family", "variant_id", "source_path", "language"])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    summary_path = out_dir / "summary.json"
    summary = {
        "out_dir": str(out_dir.resolve()),
        "family_count": len(families),
        "variants_per_family": int(args.variants_per_family),
        "total_apps": total,
        "families": families,
        "manifest_csv": str(manifest_path.resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"generated {total} apps")
    print(f"saved {manifest_path}")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
