#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ml_dataset_ir_errors.npz")
    ap.add_argument("--target", required=True, help="e.g. posit_8_0")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--eps", type=float, default=1e-30)
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=True)
    ids = data["ids"]
    y = data["Y"]

    suffix = f"|fmt{args.target}"
    mask = np.asarray([str(v).endswith(suffix) for v in ids.tolist()], dtype=bool)
    if not np.any(mask):
        raise SystemExit(f"target not found: {args.target}")

    vals = np.asarray(y[mask], dtype=np.float64)
    log_vals = np.log10(np.maximum(vals, 0.0) + float(args.eps))

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    axes[0].hist(vals, bins=int(args.bins), color="#476C9B", edgecolor="white")
    axes[0].set_title(f"{args.target} Raw Error Distribution")
    axes[0].set_xlabel("relative error")
    axes[0].set_ylabel("count")

    axes[1].hist(log_vals, bins=int(args.bins), color="#D17B0F", edgecolor="white")
    axes[1].set_title(f"{args.target} Log10 Error Distribution")
    axes[1].set_xlabel("log10(relative error + eps)")
    axes[1].set_ylabel("count")

    fig.tight_layout()
    png_path = out_prefix.with_suffix(".png")
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"count: {vals.size}")
    print(f"min: {vals.min():.6e}")
    print(f"median: {np.median(vals):.6e}")
    print(f"mean: {vals.mean():.6e}")
    print(f"p90: {np.quantile(vals, 0.9):.6e}")
    print(f"p99: {np.quantile(vals, 0.99):.6e}")
    print(f"max: {vals.max():.6e}")


if __name__ == "__main__":
    main()
