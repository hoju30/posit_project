#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_SPECS = {
    "mae": ("In-Domain MAE", "MAE", ".3f"),
    "rmse": ("In-Domain RMSE", "RMSE", ".3f"),
    "mape": ("In-Domain MAPE", "MAPE", ".3f"),
    "pick_acc": ("In-Domain Pick Accuracy", "pick_acc", ".3f"),
}

PREFERRED_ORDER = [
    "ir2vec_only",
    "ir2vec_stats",
    "ir2vec_stats_cfg",
    "full_no_quant",
    "full_quant",
]


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_order(results: dict) -> list[str]:
    return [name for name in PREFERRED_ORDER if name in results] + [
        name for name in results.keys() if name not in PREFERRED_ORDER
    ]


def metric_values(results: dict, feat_names: list[str], metric: str) -> list[float]:
    vals: list[float] = []
    for name in feat_names:
        vals.append(float(results[name]["in_domain"]["metrics"][metric]))
    return vals


def annotate(ax, bars, fmt: str) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            format(height, fmt),
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", default="mae,rmse,mape,pick_acc")
    args = ap.parse_args()

    summary = load_summary(Path(args.summary))
    results = summary["results"]
    feat_names = feature_order(results)
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not feat_names:
        raise SystemExit("no feature sets found in summary")
    for name in metric_names:
        if name not in METRIC_SPECS:
            raise SystemExit(f"unsupported metric: {name}")

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 4.8))
    if len(metric_names) == 1:
        axes = [axes]

    colors = ["#476C9B", "#468C98", "#D17B0F", "#8E5572", "#4C956C"]
    for idx, metric in enumerate(metric_names):
        title, ylabel, fmt = METRIC_SPECS[metric]
        vals = metric_values(results, feat_names, metric)
        ax = axes[idx]
        bars = ax.bar(feat_names, vals, color=colors[: len(feat_names)])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=18)
        annotate(ax, bars, fmt)

    fig.suptitle("Current Model In-Domain Feature Ablation", fontsize=14)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
