#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eval_in_domain_xgb import build_sample_views, evaluate_one, metric_dict
from train_ir_error_model import make_in_domain_split


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EPS = 1e-30


DEFAULT_CANDIDATES = [
    {
        "name": "xgb_base",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_weight": 1.0,
    },
    {
        "name": "xgb_mape_1",
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "min_child_weight": 3.0,
    },
    {
        "name": "xgb_mape_2",
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_lambda": 3.0,
        "min_child_weight": 5.0,
    },
    {
        "name": "xgb_mape_3",
        "n_estimators": 250,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 5.0,
        "min_child_weight": 8.0,
    },
    {
        "name": "xgb_mape_4",
        "n_estimators": 600,
        "max_depth": 4,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 4.0,
        "min_child_weight": 5.0,
    },
]


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "min_child_weight",
        "mae",
        "rmse",
        "mape",
        "pick_acc",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR / "ml_dataset_ir_errors.npz"))
    ap.add_argument("--out-dir", default=str(DATA_DIR / "tune_xgb_in_domain"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--rf-n-estimators", type=int, default=120)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=8)
    ap.add_argument("--rf-max-depth", type=int, default=20)
    ap.add_argument("--rf-max-samples", type=float, default=0.2)
    ap.add_argument("--calibration-ratio", type=float, default=0.15)
    ap.add_argument("--calibration-select-metric", default="mape", choices=["mae", "rmse", "mape"])
    ap.add_argument("--sort-by", default="mape", choices=["mae", "rmse", "mape", "pick_acc"])
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]
    ids = data["ids"]
    format_names = data["format_names"].tolist()

    views = build_sample_views(X, Y, ids, format_names)
    sample_kernels = views["sample_kernels"]
    row_sample_idx = views["row_sample_idx"]
    row_format_idx = views["row_format_idx"]
    row_formats = views["row_formats"]
    y_sample = views["y_sample"]

    train_samples, test_samples = make_in_domain_split(
        sample_kernels,
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )
    y_true = y_sample[test_samples]

    rf_pred, rf_extra = evaluate_one(
        "rf",
        X,
        Y,
        row_formats,
        row_sample_idx,
        row_format_idx,
        format_names,
        train_samples,
        test_samples,
        sample_kernels,
        seed=int(args.seed),
        rf_n_estimators=int(args.rf_n_estimators),
        rf_min_samples_leaf=int(args.rf_min_samples_leaf),
        rf_max_depth=int(args.rf_max_depth),
        rf_max_samples=float(args.rf_max_samples),
        xgb_n_estimators=DEFAULT_CANDIDATES[0]["n_estimators"],
        xgb_max_depth=DEFAULT_CANDIDATES[0]["max_depth"],
        xgb_learning_rate=DEFAULT_CANDIDATES[0]["learning_rate"],
        xgb_subsample=DEFAULT_CANDIDATES[0]["subsample"],
        xgb_colsample_bytree=DEFAULT_CANDIDATES[0]["colsample_bytree"],
        xgb_reg_lambda=DEFAULT_CANDIDATES[0]["reg_lambda"],
        xgb_min_child_weight=DEFAULT_CANDIDATES[0]["min_child_weight"],
        calibration_ratio=float(args.calibration_ratio),
        calibration_select_metric=str(args.calibration_select_metric),
    )
    rf_metrics = metric_dict(y_true, rf_pred, EPS)

    rows: list[dict[str, object]] = []
    runs: dict[str, object] = {
        "random_forest_baseline": {
            "metrics": rf_metrics,
            **rf_extra,
        }
    }

    for cand in DEFAULT_CANDIDATES:
        pred, extra = evaluate_one(
            "xgb",
            X,
            Y,
            row_formats,
            row_sample_idx,
            row_format_idx,
            format_names,
            train_samples,
            test_samples,
            sample_kernels,
            seed=int(args.seed),
            rf_n_estimators=int(args.rf_n_estimators),
            rf_min_samples_leaf=int(args.rf_min_samples_leaf),
            rf_max_depth=int(args.rf_max_depth),
            rf_max_samples=float(args.rf_max_samples),
            xgb_n_estimators=int(cand["n_estimators"]),
            xgb_max_depth=int(cand["max_depth"]),
            xgb_learning_rate=float(cand["learning_rate"]),
            xgb_subsample=float(cand["subsample"]),
            xgb_colsample_bytree=float(cand["colsample_bytree"]),
            xgb_reg_lambda=float(cand["reg_lambda"]),
            xgb_min_child_weight=float(cand["min_child_weight"]),
            calibration_ratio=float(args.calibration_ratio),
            calibration_select_metric=str(args.calibration_select_metric),
        )
        metrics = metric_dict(y_true, pred, EPS)
        row = {**cand, **metrics}
        rows.append(row)
        runs[str(cand["name"])] = {
            "params": cand,
            "metrics": metrics,
            **extra,
        }

    reverse = args.sort_by == "pick_acc"
    rows_sorted = sorted(rows, key=lambda r: float(r[str(args.sort_by)]), reverse=reverse)
    best = rows_sorted[0] if rows_sorted else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "candidates.csv", rows_sorted)
    summary = {
        "data": str(Path(args.data).resolve()),
        "seed": int(args.seed),
        "test_ratio": float(args.test_ratio),
        "calibration_ratio": float(args.calibration_ratio),
        "calibration_select_metric": str(args.calibration_select_metric),
        "sort_by": str(args.sort_by),
        "random_forest_baseline": {"metrics": rf_metrics, **rf_extra},
        "best_candidate": best,
        "runs": runs,
    }
    (out_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    print(f"saved {out_dir / 'summary.json'}")
    print(f"saved {out_dir / 'candidates.csv'}")
    if best is not None:
        print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
