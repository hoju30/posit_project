#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from error_feature_utils import DERIVED_STAT_FIELDS, RAW_STAT_FIELDS
from train_ir_error_model import (
    apply_log_calibrators,
    fit_log_calibrators,
    make_in_domain_split,
    parse_format_name_from_id,
    parse_kernel,
    parse_sample_prefix,
    predict_with_models,
    regroup_predictions,
    selective_calibration_summary,
    train_format_models,
)


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EPS = 1e-30


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


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err * err)))
    mape = float(np.mean(abs_err / (np.abs(y_true) + eps)))

    true_best = np.argmin(y_true, axis=1)
    pred_best = np.argmin(y_pred, axis=1)
    pick_acc = float(np.mean(true_best == pred_best))

    rows = np.arange(y_true.shape[0])
    oracle = y_true[rows, true_best]
    chosen = y_true[rows, pred_best]
    regret = chosen - oracle

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "pick_acc": pick_acc,
        "mean_regret": float(np.mean(regret)),
        "median_regret": float(np.median(regret)),
    }


def per_target_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    feature_set: str,
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
                "feature_set": feature_set,
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
        w = csv.DictWriter(f, fieldnames=["feature_set", "target", "mae", "rmse", "mape"])
        w.writeheader()
        for row in rows:
            w.writerow(row)


def build_group_indices(ids: np.ndarray, format_names: list[str]) -> dict[str, np.ndarray]:
    sample_prefixes = np.asarray([parse_sample_prefix(v) for v in ids.tolist()], dtype=object)
    unique_prefixes = np.unique(sample_prefixes)
    sample_to_index = {prefix: i for i, prefix in enumerate(unique_prefixes.tolist())}
    format_to_index = {name: i for i, name in enumerate(format_names)}

    row_sample_idx = np.asarray([sample_to_index[p] for p in sample_prefixes.tolist()], dtype=np.int64)
    row_format_idx = np.asarray(
        [format_to_index[parse_format_name_from_id(v)] for v in ids.tolist()],
        dtype=np.int64,
    )
    sample_kernels = np.asarray([parse_kernel(p) for p in unique_prefixes.tolist()], dtype=object)

    return {
        "sample_prefixes": sample_prefixes,
        "unique_prefixes": unique_prefixes,
        "row_sample_idx": row_sample_idx,
        "row_format_idx": row_format_idx,
        "sample_kernels": sample_kernels,
    }


def build_sample_targets(
    Y: np.ndarray,
    row_sample_idx: np.ndarray,
    row_format_idx: np.ndarray,
    n_samples: int,
    n_formats: int,
) -> np.ndarray:
    y_out = np.full((n_samples, n_formats), np.nan, dtype=np.float32)
    for row_idx, y in enumerate(Y.tolist()):
        y_out[int(row_sample_idx[row_idx]), int(row_format_idx[row_idx])] = float(y)
    if np.isnan(y_out).any():
        raise RuntimeError("sample-level target matrix contains NaN")
    return y_out


def feature_index_map(meta: dict[str, object]) -> dict[str, np.ndarray]:
    ir_dim = int(meta.get("ir_dim", 300))
    shared_feat_dim = int(meta.get("shared_feat_dim", 0))
    format_feat_dim = int(meta.get("format_feat_dim", 0))
    stats_dim = len(meta.get("stats_feat_names", []))
    cfg_dim = int(meta.get("cfg_dim", 0) or 0)
    quant_dim = len(meta.get("quant_feat_names", []))
    scalar_dim = shared_feat_dim - stats_dim - cfg_dim
    if scalar_dim < 0:
        raise ValueError("invalid meta: scalar_dim < 0")
    fmt_desc_dim = format_feat_dim - quant_dim
    if fmt_desc_dim < 0:
        raise ValueError("invalid meta: fmt_desc_dim < 0")
    raw_stats_dim = 2 * len(RAW_STAT_FIELDS)
    derived_stats_dim = 2 * len(DERIVED_STAT_FIELDS)
    if stats_dim < raw_stats_dim + derived_stats_dim:
        raise ValueError("invalid meta: stats_dim smaller than raw+derived stats")

    ir_end = ir_dim
    scalar_end = ir_end + scalar_dim
    raw_stats_end = scalar_end + raw_stats_dim
    stats_end = scalar_end + stats_dim
    cfg_end = stats_end + cfg_dim
    fmt_desc_end = cfg_end + fmt_desc_dim
    all_end = fmt_desc_end + quant_dim

    return {
        "ir2vec_only": np.arange(0, ir_end, dtype=np.int64),
        "ir2vec_stats": np.arange(0, stats_end, dtype=np.int64),
        "raw_only": np.concatenate(
            [
                np.arange(0, raw_stats_end, dtype=np.int64),
                np.arange(cfg_end, all_end, dtype=np.int64),
            ]
        ),
        "raw_plus_derived": np.arange(0, all_end, dtype=np.int64),
        "full_no_quant": np.arange(0, fmt_desc_end, dtype=np.int64),
        "full_quant": np.arange(0, all_end, dtype=np.int64),
    }


def evaluate_feature_set(
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
    n_estimators: int,
    min_samples_leaf: int,
    max_depth: int,
    max_samples: float,
    random_state: int,
    calibration_ratio: float,
    calibration_select_metric: str,
) -> tuple[np.ndarray, dict[str, object]]:
    train_set = set(train_samples.tolist())
    row_train_mask = np.asarray([idx in train_set for idx in row_sample_idx.tolist()], dtype=bool)
    row_test_mask = ~row_train_mask

    inner_fit_local, inner_cal_local = make_in_domain_split(
        sample_kernels[train_samples],
        test_ratio=float(calibration_ratio),
        seed=int(random_state) + 17,
    )
    inner_fit_samples = train_samples[inner_fit_local]
    inner_cal_samples = train_samples[inner_cal_local]
    fit_set = set(inner_fit_samples.tolist())
    cal_set = set(inner_cal_samples.tolist())
    row_fit_mask = np.asarray([idx in fit_set for idx in row_sample_idx.tolist()], dtype=bool)
    row_cal_mask = np.asarray([idx in cal_set for idx in row_sample_idx.tolist()], dtype=bool)

    cal_models, _ = train_format_models(
        X[row_fit_mask],
        Y[row_fit_mask],
        row_formats[row_fit_mask],
        format_names,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_samples=max_samples,
        random_state=random_state,
        eps=EPS,
    )
    pred_cal_rows = predict_with_models(
        cal_models,
        X[row_cal_mask],
        row_formats[row_cal_mask],
        format_names,
        eps=EPS,
    )
    pred_cal_base = regroup_predictions(
        pred_cal_rows,
        np.where(row_cal_mask)[0],
        inner_cal_samples,
        row_sample_idx,
        row_format_idx,
        len(format_names),
    )

    y_cal_true = np.full((inner_cal_samples.size, len(format_names)), np.nan, dtype=np.float32)
    cal_sample_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(inner_cal_samples.tolist())}
    for row_idx in np.where(row_cal_mask)[0].tolist():
        local_idx = cal_sample_to_local[int(row_sample_idx[row_idx])]
        fmt_idx = int(row_format_idx[row_idx])
        y_cal_true[local_idx, fmt_idx] = float(Y[row_idx])
    if np.isnan(y_cal_true).any():
        raise RuntimeError("calibration target regroup failed")

    slopes, intercepts, calibration_summary = fit_log_calibrators(y_cal_true, pred_cal_base, format_names, EPS)
    pred_cal_calibrated = apply_log_calibrators(pred_cal_base, slopes, intercepts, EPS)
    selective_summary = selective_calibration_summary(
        y_cal_true,
        pred_cal_base,
        pred_cal_calibrated,
        format_names,
        metric=calibration_select_metric,
        eps=EPS,
    )

    final_models, _ = train_format_models(
        X[row_train_mask],
        Y[row_train_mask],
        row_formats[row_train_mask],
        format_names,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_samples=max_samples,
        random_state=random_state,
        eps=EPS,
    )
    pred_test_rows = predict_with_models(
        final_models,
        X[row_test_mask],
        row_formats[row_test_mask],
        format_names,
        eps=EPS,
    )
    pred_test = regroup_predictions(
        pred_test_rows,
        np.where(row_test_mask)[0],
        test_samples,
        row_sample_idx,
        row_format_idx,
        len(format_names),
    )
    pred_test_cal = apply_log_calibrators(pred_test, slopes, intercepts, EPS)
    pred_test_selective = pred_test.copy()
    for j, name in enumerate(format_names):
        if bool(selective_summary[str(name)]["use_calibrated"]):
            pred_test_selective[:, j] = pred_test_cal[:, j]

    summary = {
        "x_dim": int(X.shape[1]),
        "calibration": {
            "metric": calibration_select_metric,
            "selected_formats": sorted(
                [name for name, info in selective_summary.items() if bool(info["use_calibrated"])]
            ),
            "selected_count": int(
                sum(1 for info in selective_summary.values() if bool(info["use_calibrated"]))
            ),
            "selective_summary": selective_summary,
            "raw_summary": calibration_summary,
        },
    }
    return pred_test_selective, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR / "ml_dataset_ir_errors.npz"))
    ap.add_argument("--out-dir", default=str(DATA_DIR / "eval_feature_ablation_current"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--n-estimators", type=int, default=120)
    ap.add_argument("--min-samples-leaf", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=20)
    ap.add_argument("--max-samples", type=float, default=0.2)
    ap.add_argument("--calibration-ratio", type=float, default=0.15)
    ap.add_argument("--calibration-select-metric", default="mape", choices=["mae", "rmse", "mape"])
    ap.add_argument(
        "--feature-sets",
        default="all",
        help="comma-separated: ir2vec_only,ir2vec_stats,raw_only,raw_plus_derived,full_no_quant,full_quant or 'all'",
    )
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=True)
    X_all = data["X"]
    Y = data["Y"]
    ids = data["ids"]
    format_names = data["format_names"].tolist()
    meta = dict(data["meta"].item())

    group_info = build_group_indices(ids, format_names)
    row_sample_idx = group_info["row_sample_idx"]
    row_format_idx = group_info["row_format_idx"]
    sample_kernels = group_info["sample_kernels"]
    row_formats = np.asarray([parse_format_name_from_id(v) for v in ids.tolist()], dtype=object)
    y_sample = build_sample_targets(Y, row_sample_idx, row_format_idx, len(group_info["unique_prefixes"]), len(format_names))

    feature_sets_map = feature_index_map(meta)
    if args.feature_sets.strip().lower() == "all":
        feature_names = list(feature_sets_map.keys())
    else:
        feature_names = [s.strip() for s in args.feature_sets.split(",") if s.strip()]
        for name in feature_names:
            if name not in feature_sets_map:
                raise SystemExit(f"unsupported feature set: {name}")

    train_samples, test_samples = make_in_domain_split(
        sample_kernels,
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )
    y_true = y_sample[test_samples]

    results: dict[str, object] = {}
    per_target_all: list[dict[str, float | str]] = []

    for feat_name in feature_names:
        cols = feature_sets_map[feat_name]
        x_use = X_all[:, cols]
        pred, extra = evaluate_feature_set(
            x_use,
            Y,
            row_formats,
            row_sample_idx,
            row_format_idx,
            format_names,
            train_samples,
            test_samples,
            sample_kernels,
            n_estimators=int(args.n_estimators),
            min_samples_leaf=int(args.min_samples_leaf),
            max_depth=int(args.max_depth),
            max_samples=float(args.max_samples),
            random_state=int(args.seed),
            calibration_ratio=float(args.calibration_ratio),
            calibration_select_metric=str(args.calibration_select_metric),
        )
        metrics = metric_dict(y_true, pred, EPS)
        results[feat_name] = {
            "in_domain": {
                "metrics": metrics,
                **extra,
            }
        }
        per_target_all.extend(per_target_rows(y_true, pred, format_names, feat_name, EPS))

    summary = {
        "data": str(Path(args.data).resolve()),
        "n_rows": int(X_all.shape[0]),
        "n_samples": int(len(group_info["unique_prefixes"])),
        "n_formats": int(len(format_names)),
        "format_names": format_names,
        "seed": int(args.seed),
        "test_ratio": float(args.test_ratio),
        "calibration_ratio": float(args.calibration_ratio),
        "calibration_select_metric": str(args.calibration_select_metric),
        "feature_meta": meta,
        "results": results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    write_per_target_csv(out_dir / "per_target.csv", per_target_all)
    print(f"saved {out_dir / 'summary.json'}")
    print(f"saved {out_dir / 'per_target.csv'}")
    print(json.dumps({k: v["in_domain"]["metrics"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
