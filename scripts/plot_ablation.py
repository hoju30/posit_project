#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

METRIC_SPECS = {
    "pick_acc": ("In-Domain Pick Accuracy", "pick_acc", ".3f", False),
    "mape": ("In-Domain MAPE", "mape", ".3f", False),
    "mae": ("In-Domain MAE", "mae", ".3f", False),
    "rmse": ("In-Domain RMSE", "rmse", ".3f", False),
}


def load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_feature_order(results: dict) -> list[str]:
    preferred = ["ir2vec_only", "ir2vec_stats", "ir2vec_stats_cfg"]
    return [k for k in preferred if k in results] + [
        k for k in results.keys() if k not in preferred
    ]


def metric_list(results: dict, feature_order: list[str], domain: str, metric: str, ood_kernel: str | None = None) -> list[float]:
    vals: list[float] = []
    for feat in feature_order:
        if domain == "in_domain":
            vals.append(float(results[feat]["in_domain"]["metrics"][metric]))
        else:
            if ood_kernel is None:
                raise ValueError("ood_kernel is required for ood metrics")
            vals.append(float(results[feat]["ood"][ood_kernel]["metrics"][metric]))
    return vals


def annotate_bars(ax, bars, fmt: str) -> None:
    for b in bars:
        h = b.get_height()
        ax.annotate(
            format(h, fmt),
            xy=(b.get_x() + b.get_width() / 2.0, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="data/eval_ablation/summary.json")
    ap.add_argument("--out", default="data/eval_ablation/ablation_in_domain.png")
    ap.add_argument(
        "--metrics",
        default="pick_acc,mape",
        help="comma-separated in-domain metrics, e.g. mae,rmse",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary)
    out_path = Path(args.out)
    obj = load_summary(summary_path)
    results = obj["results"]
    feature_order = pick_feature_order(results)
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]

    if not feature_order:
        raise SystemExit("no ablation results found")
    if not metric_names:
        raise SystemExit("no metrics selected")
    for metric in metric_names:
        if metric not in METRIC_SPECS:
            raise SystemExit(f"unsupported metric: {metric}")

    labels = feature_order

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 4.8))
    fig.suptitle("In-Domain Ablation Summary", fontsize=14)
    if len(metric_names) == 1:
        axes = [axes]

    plots = []
    for idx, metric in enumerate(metric_names):
        vals = metric_list(results, feature_order, "in_domain", metric)
        title, ylabel, fmt, use_log = METRIC_SPECS[metric]
        plots.append((axes[idx], vals, title, ylabel, fmt, use_log))

    colors = ["#4C956C", "#2C6E91", "#D17B0F", "#8E5572"]
    for i, (ax, vals, title, ylabel, fmt, use_log) in enumerate(plots):
        bars = ax.bar(labels, vals, color=colors[i % len(colors)])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if use_log:
            ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=15)
        annotate_bars(ax, bars, fmt)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
