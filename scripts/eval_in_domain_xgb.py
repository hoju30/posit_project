#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_ir_error_model import (
    apply_log_calibrators,
    fit_log_calibrators,
    make_in_domain_split,
    parse_format_name_from_id,
    parse_kernel,
    parse_sample_prefix,
    regroup_predictions,
    selective_calibration_summary,
    train_format_models,
    predict_with_models,
)
from train_ir_error_model_xgb import train_format_models_xgb, predict_with_models_xgb


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EPS = 1e-30


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err * err)))
    mape = float(np.mean(abs_err / (np.abs(y_true) + eps)))

    true_best = np.argmin(y_true, axis=1)
    pred_best = np.argmin(y_pred, axis=1)
    pick_acc = float(np.mean(true_best == pred_best))
    return {"mae": mae, "rmse": rmse, "mape": mape, "pick_acc": pick_acc}


def per_target_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    model_name: str,
    eps: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for j, target in enumerate(target_names):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        err = yp - yt
        abs_err = np.abs(err)
        rows.append(
            {
                "model": model_name,
                "target": target,
                "mae": float(np.mean(abs_err)),
                "rmse": float(np.sqrt(np.mean(err * err))),
                "mape": float(np.mean(abs_err / (np.abs(yt) + eps))),
            }
        )
    return rows


def write_per_target_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "target", "mae", "rmse", "mape"])
        w.writeheader()
        for row in rows:
            w.writerow(row)


def build_sample_views(
    X: np.ndarray,
    Y: np.ndarray,
    ids: np.ndarray,
    format_names: list[str],
) -> dict[str, object]:
    sample_prefixes = np.asarray([parse_sample_prefix(v) for v in ids], dtype=object)
    unique_prefixes = np.unique(sample_prefixes)
    sample_to_index = {prefix: i for i, prefix in enumerate(unique_prefixes.tolist())}
    format_to_index = {name: i for i, name in enumerate(format_names)}

    n_samples = unique_prefixes.size
    n_formats = len(format_names)
    y_sample = np.full((n_samples, n_formats), np.nan, dtype=np.float32)
    row_sample_idx = np.zeros((ids.shape[0],), dtype=np.int64)
    row_format_idx = np.zeros((ids.shape[0],), dtype=np.int64)

    for row_idx, raw_id in enumerate(ids.tolist()):
        sid = str(raw_id)
        prefix = parse_sample_prefix(sid)
        sample_idx = sample_to_index[prefix]
        fmt_name = parse_format_name_from_id(sid)
        fmt_idx = format_to_index[fmt_name]
        row_sample_idx[row_idx] = sample_idx
        row_format_idx[row_idx] = fmt_idx
        y_sample[sample_idx, fmt_idx] = Y[row_idx]

    if np.isnan(y_sample).any():
        raise RuntimeError("sample-level target matrix contains NaN")

    sample_kernels = np.asarray([parse_kernel(p) for p in unique_prefixes.tolist()], dtype=object)
    row_formats = np.asarray([parse_format_name_from_id(v) for v in ids.tolist()], dtype=object)
    return {
        "sample_kernels": sample_kernels,
        "row_sample_idx": row_sample_idx,
        "row_format_idx": row_format_idx,
        "row_formats": row_formats,
        "y_sample": y_sample,
    }


def evaluate_one(
    kind: str,
    X: np.ndarray,
    Y: np.ndarray,
    row_formats: np.ndarray,
    row_sample_idx: np.ndarray,
    row_format_idx: np.ndarray,
    format_names: list[str],
    train_samples: np.ndarray,
    test_samples: np.ndarray,
    sample_kernels: np.ndarray,
    *,
    seed: int,
    rf_n_estimators: int,
    rf_min_samples_leaf: int,
    rf_max_depth: int,
    rf_max_samples: float,
    xgb_n_estimators: int,
    xgb_max_depth: int,
    xgb_learning_rate: float,
    xgb_subsample: float,
    xgb_colsample_bytree: float,
    xgb_reg_lambda: float,
    xgb_min_child_weight: float,
    calibration_ratio: float,
    calibration_select_metric: str,
) -> tuple[np.ndarray, dict[str, object]]:
    train_set = set(train_samples.tolist())
    row_train_mask = np.asarray([idx in train_set for idx in row_sample_idx.tolist()], dtype=bool)
    row_test_mask = ~row_train_mask

    fit_local, cal_local = make_in_domain_split(
        sample_kernels[train_samples],
        test_ratio=float(calibration_ratio),
        seed=int(seed) + 17,
    )
    fit_samples = train_samples[fit_local]
    cal_samples = train_samples[cal_local]
    fit_set = set(fit_samples.tolist())
    cal_set = set(cal_samples.tolist())
    row_fit_mask = np.asarray([idx in fit_set for idx in row_sample_idx.tolist()], dtype=bool)
    row_cal_mask = np.asarray([idx in cal_set for idx in row_sample_idx.tolist()], dtype=bool)

    if kind == "rf":
        cal_models, _ = train_format_models(
            X[row_fit_mask],
            Y[row_fit_mask],
            row_formats[row_fit_mask],
            format_names,
            n_estimators=rf_n_estimators,
            min_samples_leaf=rf_min_samples_leaf,
            max_depth=rf_max_depth,
            max_samples=rf_max_samples,
            random_state=seed,
            eps=EPS,
        )
        pred_cal_rows = predict_with_models(
            cal_models,
            X[row_cal_mask],
            row_formats[row_cal_mask],
            format_names,
            eps=EPS,
        )
        final_models, _ = train_format_models(
            X[row_train_mask],
            Y[row_train_mask],
            row_formats[row_train_mask],
            format_names,
            n_estimators=rf_n_estimators,
            min_samples_leaf=rf_min_samples_leaf,
            max_depth=rf_max_depth,
            max_samples=rf_max_samples,
            random_state=seed,
            eps=EPS,
        )
        pred_test_rows = predict_with_models(
            final_models,
            X[row_test_mask],
            row_formats[row_test_mask],
            format_names,
            eps=EPS,
        )
    elif kind == "xgb":
        cal_models, _ = train_format_models_xgb(
            X[row_fit_mask],
            Y[row_fit_mask],
            row_formats[row_fit_mask],
            format_names,
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            subsample=xgb_subsample,
            colsample_bytree=xgb_colsample_bytree,
            reg_lambda=xgb_reg_lambda,
            min_child_weight=xgb_min_child_weight,
            random_state=seed,
            eps=EPS,
        )
        pred_cal_rows = predict_with_models_xgb(
            cal_models,
            X[row_cal_mask],
            row_formats[row_cal_mask],
            format_names,
            eps=EPS,
        )
        final_models, _ = train_format_models_xgb(
            X[row_train_mask],
            Y[row_train_mask],
            row_formats[row_train_mask],
            format_names,
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            subsample=xgb_subsample,
            colsample_bytree=xgb_colsample_bytree,
            reg_lambda=xgb_reg_lambda,
            min_child_weight=xgb_min_child_weight,
            random_state=seed,
            eps=EPS,
        )
        pred_test_rows = predict_with_models_xgb(
            final_models,
            X[row_test_mask],
            row_formats[row_test_mask],
            format_names,
            eps=EPS,
        )
    else:
        raise ValueError(f"unsupported kind: {kind}")

    pred_cal = regroup_predictions(
        pred_cal_rows,
        np.where(row_cal_mask)[0],
        cal_samples,
        row_sample_idx,
        row_format_idx,
        len(format_names),
    )
    pred_test = regroup_predictions(
        pred_test_rows,
        np.where(row_test_mask)[0],
        test_samples,
        row_sample_idx,
        row_format_idx,
        len(format_names),
    )

    y_cal_true = np.full((cal_samples.size, len(format_names)), np.nan, dtype=np.float32)
    cal_sample_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(cal_samples.tolist())}
    for row_idx in np.where(row_cal_mask)[0].tolist():
        sample_idx = int(row_sample_idx[row_idx])
        local_idx = cal_sample_to_local[sample_idx]
        fmt_idx = int(row_format_idx[row_idx])
        y_cal_true[local_idx, fmt_idx] = float(Y[row_idx])
    if np.isnan(y_cal_true).any():
        raise RuntimeError("calibration target regroup failed")

    slopes, intercepts, calibration_summary = fit_log_calibrators(y_cal_true, pred_cal, format_names, EPS)
    pred_cal_calibrated = apply_log_calibrators(pred_cal, slopes, intercepts, EPS)
    selective_summary = selective_calibration_summary(
        y_cal_true,
        pred_cal,
        pred_cal_calibrated,
        format_names,
        metric=calibration_select_metric,
        eps=EPS,
    )
    pred_test_cal = apply_log_calibrators(pred_test, slopes, intercepts, EPS)
    pred_test_selective = pred_test.copy()
    for j, name in enumerate(format_names):
        if bool(selective_summary[str(name)]["use_calibrated"]):
            pred_test_selective[:, j] = pred_test_cal[:, j]

    return pred_test_selective, {
        "selected_formats": sorted(
            [name for name, info in selective_summary.items() if bool(info["use_calibrated"])]
        ),
        "selected_count": int(sum(1 for info in selective_summary.values() if bool(info["use_calibrated"]))),
        "selective_summary": selective_summary,
        "calibration_summary": calibration_summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR / "ml_dataset_ir_errors.npz"))
    ap.add_argument("--out-dir", default=str(DATA_DIR / "eval_in_domain_xgb"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--rf-n-estimators", type=int, default=120)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=8)
    ap.add_argument("--rf-max-depth", type=int, default=20)
    ap.add_argument("--rf-max-samples", type=float, default=0.2)
    ap.add_argument("--xgb-n-estimators", type=int, default=300)
    ap.add_argument("--xgb-max-depth", type=int, default=6)
    ap.add_argument("--xgb-learning-rate", type=float, default=0.05)
    ap.add_argument("--xgb-subsample", type=float, default=0.8)
    ap.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    ap.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    ap.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    ap.add_argument("--calibration-ratio", type=float, default=0.15)
    ap.add_argument("--calibration-select-metric", default="mape", choices=["mae", "rmse", "mape"])
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

    pred_rf, rf_extra = evaluate_one(
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
        xgb_n_estimators=int(args.xgb_n_estimators),
        xgb_max_depth=int(args.xgb_max_depth),
        xgb_learning_rate=float(args.xgb_learning_rate),
        xgb_subsample=float(args.xgb_subsample),
        xgb_colsample_bytree=float(args.xgb_colsample_bytree),
        xgb_reg_lambda=float(args.xgb_reg_lambda),
        xgb_min_child_weight=float(args.xgb_min_child_weight),
        calibration_ratio=float(args.calibration_ratio),
        calibration_select_metric=str(args.calibration_select_metric),
    )
    pred_xgb, xgb_extra = evaluate_one(
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
        xgb_n_estimators=int(args.xgb_n_estimators),
        xgb_max_depth=int(args.xgb_max_depth),
        xgb_learning_rate=float(args.xgb_learning_rate),
        xgb_subsample=float(args.xgb_subsample),
        xgb_colsample_bytree=float(args.xgb_colsample_bytree),
        xgb_reg_lambda=float(args.xgb_reg_lambda),
        xgb_min_child_weight=float(args.xgb_min_child_weight),
        calibration_ratio=float(args.calibration_ratio),
        calibration_select_metric=str(args.calibration_select_metric),
    )

    results = {
        "random_forest_selective_calibrated": {
            "metrics": metric_dict(y_true, pred_rf, EPS),
            **rf_extra,
        },
        "xgboost_selective_calibrated": {
            "metrics": metric_dict(y_true, pred_xgb, EPS),
            **xgb_extra,
        },
    }
    deltas = {
        k: float(results["xgboost_selective_calibrated"]["metrics"][k] - results["random_forest_selective_calibrated"]["metrics"][k])
        for k in results["random_forest_selective_calibrated"]["metrics"]
    }

    per_target = []
    per_target.extend(per_target_rows(y_true, pred_rf, format_names, "random_forest_selective_calibrated", EPS))
    per_target.extend(per_target_rows(y_true, pred_xgb, format_names, "xgboost_selective_calibrated", EPS))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "data": str(Path(args.data).resolve()),
        "n_rows": int(X.shape[0]),
        "n_samples": int(y_sample.shape[0]),
        "n_formats": int(len(format_names)),
        "format_names": format_names,
        "seed": int(args.seed),
        "test_ratio": float(args.test_ratio),
        "calibration_ratio": float(args.calibration_ratio),
        "results": results,
        "xgboost_minus_random_forest": deltas,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_per_target_csv(out_dir / "per_target.csv", per_target)
    print(f"saved {out_dir / 'summary.json'}")
    print(f"saved {out_dir / 'per_target.csv'}")
    print(json.dumps({k: v['metrics'] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
