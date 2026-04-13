#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
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
)

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "xgboost is not installed in the active environment. Run: pip install xgboost"
    ) from exc


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


def train_format_models_xgb(
    X: np.ndarray,
    Y: np.ndarray,
    row_formats: np.ndarray,
    format_names: list[str],
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    min_child_weight: float,
    random_state: int,
    eps: float,
) -> tuple[dict[str, XGBRegressor], dict[str, int]]:
    models: dict[str, XGBRegressor] = {}
    train_counts: dict[str, int] = {}
    for format_name in format_names:
        mask = row_formats == str(format_name)
        x_fmt = X[mask]
        y_fmt = Y[mask]
        y_log = np.log10(np.maximum(y_fmt, 0.0) + eps)
        reg = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=1,
        )
        reg.fit(x_fmt, y_log)
        models[str(format_name)] = reg
        train_counts[str(format_name)] = int(mask.sum())
    return models, train_counts


def predict_with_models_xgb(
    models: dict[str, XGBRegressor],
    X: np.ndarray,
    row_formats: np.ndarray,
    format_names: list[str],
    *,
    eps: float,
) -> np.ndarray:
    pred = np.zeros((X.shape[0],), dtype=np.float32)
    for format_name in format_names:
        mask = row_formats == str(format_name)
        if not np.any(mask):
            continue
        pred_log = models[str(format_name)].predict(X[mask])
        pred[mask] = np.maximum(np.power(10.0, pred_log) - eps, 0.0)
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR / "ml_dataset_ir_errors.npz"))
    ap.add_argument("--out", default=str(MODEL_DIR / "ir2vec_error_predictor_xgb.joblib"))
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--reg-lambda", type=float, default=4.0)
    ap.add_argument("--min-child-weight", type=float, default=5.0)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--calibration-ratio", type=float, default=0.15)
    ap.add_argument("--calibration-select-metric", default="mape", choices=["mae", "rmse", "mape"])
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]
    ids = data["ids"]
    format_names = data["format_names"].tolist()
    meta = data.get("meta", None)

    eps = 1e-30
    row_formats = np.asarray([parse_format_name_from_id(v) for v in ids.tolist()], dtype=object)
    sample_prefixes = np.asarray([parse_sample_prefix(v) for v in ids.tolist()], dtype=object)
    unique_prefixes = np.unique(sample_prefixes)
    sample_to_index = {prefix: i for i, prefix in enumerate(unique_prefixes.tolist())}
    format_to_index = {name: i for i, name in enumerate(format_names)}
    row_sample_idx = np.asarray([sample_to_index[prefix] for prefix in sample_prefixes.tolist()], dtype=np.int64)
    row_format_idx = np.asarray(
        [format_to_index[parse_format_name_from_id(v)] for v in ids.tolist()],
        dtype=np.int64,
    )
    sample_kernels = np.asarray([parse_kernel(p) for p in unique_prefixes.tolist()], dtype=object)

    tr_samples, cal_samples = make_in_domain_split(
        sample_kernels,
        test_ratio=float(args.calibration_ratio),
        seed=int(args.random_state) + 17,
    )
    tr_sample_set = set(tr_samples.tolist())
    cal_sample_set = set(cal_samples.tolist())
    row_train_mask = np.asarray([idx in tr_sample_set for idx in row_sample_idx.tolist()], dtype=bool)
    row_cal_mask = np.asarray([idx in cal_sample_set for idx in row_sample_idx.tolist()], dtype=bool)

    cal_models, _ = train_format_models_xgb(
        X[row_train_mask],
        Y[row_train_mask],
        row_formats[row_train_mask],
        format_names,
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        learning_rate=float(args.learning_rate),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        reg_lambda=float(args.reg_lambda),
        min_child_weight=float(args.min_child_weight),
        random_state=int(args.random_state),
        eps=eps,
    )
    pred_cal_rows = predict_with_models_xgb(
        cal_models,
        X[row_cal_mask],
        row_formats[row_cal_mask],
        format_names,
        eps=eps,
    )
    pred_cal_base = regroup_predictions(
        pred_cal_rows,
        np.where(row_cal_mask)[0],
        cal_samples,
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

    slopes, intercepts, calibration_summary = fit_log_calibrators(y_cal_true, pred_cal_base, format_names, eps)
    pred_cal_calibrated = apply_log_calibrators(pred_cal_base, slopes, intercepts, eps)
    selective_summary = selective_calibration_summary(
        y_cal_true,
        pred_cal_base,
        pred_cal_calibrated,
        format_names,
        metric=str(args.calibration_select_metric),
        eps=eps,
    )

    models, train_counts = train_format_models_xgb(
        X,
        Y,
        row_formats,
        format_names,
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        learning_rate=float(args.learning_rate),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        reg_lambda=float(args.reg_lambda),
        min_child_weight=float(args.min_child_weight),
        random_state=int(args.random_state),
        eps=eps,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "format_names": format_names,
            "meta": dict(meta.item()) if meta is not None else None,
            "y_transform": {"type": "log10", "eps": eps},
            "model_kind": "per_format_models_xgb",
            "train_counts": train_counts,
            "calibration": {
                "enabled": True,
                "type": "log_linear_selective",
                "ratio": float(args.calibration_ratio),
                "select_metric": str(args.calibration_select_metric),
                "formats": {
                    str(name): {
                        "slope": float(slopes[i]),
                        "intercept": float(intercepts[i]),
                        "use_calibrated": bool(selective_summary[str(name)]["use_calibrated"]),
                    }
                    for i, name in enumerate(format_names)
                },
                "summary": calibration_summary,
                "selective_summary": selective_summary,
                "calibration_samples": int(cal_samples.size),
            },
            "train_config": {
                "n_estimators": int(args.n_estimators),
                "max_depth": int(args.max_depth),
                "learning_rate": float(args.learning_rate),
                "subsample": float(args.subsample),
                "colsample_bytree": float(args.colsample_bytree),
                "reg_lambda": float(args.reg_lambda),
                "min_child_weight": float(args.min_child_weight),
                "random_state": int(args.random_state),
            },
        },
        out_path,
        compress=3,
    )

    print(f"saved {out_path}")
    print("formats:", format_names)
    print("rows:", int(X.shape[0]))
    print("features:", int(X.shape[1]))


if __name__ == "__main__":
    main()
