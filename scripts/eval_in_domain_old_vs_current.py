#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EPS = 1e-30


def parse_sample_prefix(sample_id: str) -> str:
    return str(sample_id).rsplit("|fmt", 1)[0]


def parse_kernel(prefix: str) -> str:
    return str(prefix).split("|", 1)[0]


def make_in_domain_split(kernels: np.ndarray, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for kernel in sorted(set(kernels.tolist())):
        idx = np.where(kernels == kernel)[0]
        if idx.size < 2:
            train_parts.append(idx)
            continue
        n_test = int(round(idx.size * test_ratio))
        n_test = max(1, min(n_test, idx.size - 1))
        chosen = np.asarray(rng.choice(idx, size=n_test, replace=False), dtype=np.int64)
        mask = np.ones(idx.size, dtype=bool)
        chosen_set = set(chosen.tolist())
        for i, v in enumerate(idx.tolist()):
            if v in chosen_set:
                mask[i] = False
        train_parts.append(idx[mask])
        test_parts.append(chosen)

    train_idx = np.concatenate(train_parts) if train_parts else np.zeros((0,), dtype=np.int64)
    test_idx = np.concatenate(test_parts) if test_parts else np.zeros((0,), dtype=np.int64)
    return train_idx, test_idx


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


def build_sample_level_views(
    X: np.ndarray,
    Y: np.ndarray,
    ids: np.ndarray,
    format_names: list[str],
    meta: dict[str, object],
) -> dict[str, object]:
    ir_dim = int(meta.get("ir_dim", 300))
    stats_dim = len(meta.get("stats_feat_names", []))
    quant_dim = len(meta.get("quant_feat_names", []))
    scalar_dim = int(meta.get("shared_feat_dim", 0)) - stats_dim

    stats_end = ir_dim + scalar_dim + stats_dim

    sample_prefixes = np.asarray([parse_sample_prefix(v) for v in ids], dtype=object)
    unique_prefixes = np.unique(sample_prefixes)
    sample_to_index = {prefix: i for i, prefix in enumerate(unique_prefixes.tolist())}
    format_to_index = {name: i for i, name in enumerate(format_names)}

    n_samples = unique_prefixes.size
    n_formats = len(format_names)

    X_stats = np.zeros((n_samples, stats_end), dtype=np.float32)
    Y_sample = np.full((n_samples, n_formats), np.nan, dtype=np.float32)

    row_sample_idx = np.zeros((ids.shape[0],), dtype=np.int64)
    row_format_idx = np.zeros((ids.shape[0],), dtype=np.int64)

    for row_idx, raw_id in enumerate(ids.tolist()):
        sid = str(raw_id)
        prefix = parse_sample_prefix(sid)
        sample_idx = sample_to_index[prefix]
        fmt_name = sid.rsplit("|fmt", 1)[1]
        fmt_idx = format_to_index[fmt_name]

        row_sample_idx[row_idx] = sample_idx
        row_format_idx[row_idx] = fmt_idx
        X_stats[sample_idx] = X[row_idx, :stats_end]
        Y_sample[sample_idx, fmt_idx] = Y[row_idx]

    if np.isnan(Y_sample).any():
        raise RuntimeError("sample-level target matrix contains NaN; dataset expansion is incomplete")

    sample_kernels = np.asarray([parse_kernel(p) for p in unique_prefixes.tolist()], dtype=object)

    return {
        "sample_ids": unique_prefixes,
        "sample_kernels": sample_kernels,
        "X_stats": X_stats,
        "Y_sample": Y_sample,
        "row_sample_idx": row_sample_idx,
        "row_format_idx": row_format_idx,
        "stats_end": stats_end,
        "quant_dim": quant_dim,
    }


def fit_predict_multioutput(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_depth: int,
    max_samples: float,
) -> np.ndarray:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_features="sqrt",
        max_samples=max_samples,
        random_state=seed,
        n_jobs=-1,
    )
    y_log = np.log10(np.maximum(Y_train, 0.0) + EPS)
    model.fit(X_train, y_log)
    pred_log = model.predict(X_test)
    pred = np.power(10.0, pred_log) - EPS
    return np.maximum(pred, 0.0)


def train_per_format_models(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    fmt_train: np.ndarray,
    format_names: list[str],
    seed: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_depth: int,
    max_samples: float,
) -> dict[int, RandomForestRegressor]:
    models: dict[int, RandomForestRegressor] = {}
    for fmt_idx, _ in enumerate(format_names):
        tr_mask = fmt_train == fmt_idx
        if not np.any(tr_mask):
            continue
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_features="sqrt",
            max_samples=max_samples,
            random_state=seed,
            n_jobs=-1,
        )
        y_log = np.log10(np.maximum(Y_train[tr_mask], 0.0) + EPS)
        model.fit(X_train[tr_mask], y_log)
        models[fmt_idx] = model
    return models


def predict_per_format_models(
    models: dict[int, RandomForestRegressor],
    X_test: np.ndarray,
    fmt_test: np.ndarray,
    format_names: list[str],
) -> np.ndarray:
    pred = np.zeros((X_test.shape[0],), dtype=np.float32)
    for fmt_idx, _ in enumerate(format_names):
        te_mask = fmt_test == fmt_idx
        if not np.any(te_mask):
            continue
        model = models[fmt_idx]
        pred_log = model.predict(X_test[te_mask])
        pred[te_mask] = np.maximum(np.power(10.0, pred_log) - EPS, 0.0)
    return pred


def regroup_row_predictions(
    pred_rows: np.ndarray,
    row_indices: np.ndarray,
    target_samples: np.ndarray,
    row_sample_idx: np.ndarray,
    ids: np.ndarray,
    format_names: list[str],
) -> np.ndarray:
    pred = np.full((target_samples.size, len(format_names)), np.nan, dtype=np.float32)
    sample_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(target_samples.tolist())}
    format_to_index = {name: i for i, name in enumerate(format_names)}
    for row_idx, val in zip(row_indices.tolist(), pred_rows.tolist()):
        fmt = str(ids[row_idx]).rsplit("|fmt", 1)[1]
        global_sample_idx = int(row_sample_idx[row_idx])
        local_sample_idx = sample_to_local[global_sample_idx]
        pred[local_sample_idx, format_to_index[fmt]] = float(val)
    if np.isnan(pred).any():
        raise RuntimeError("row predictions are incomplete after regrouping")
    return pred


def fit_log_calibrators(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    format_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    slopes = np.ones((len(format_names),), dtype=np.float64)
    intercepts = np.zeros((len(format_names),), dtype=np.float64)
    summary: dict[str, dict[str, float]] = {}
    for j, name in enumerate(format_names):
        x = np.log10(np.maximum(y_pred[:, j], 0.0) + EPS)
        y = np.log10(np.maximum(y_true[:, j], 0.0) + EPS)
        if x.size < 2:
            a, b = 1.0, 0.0
        elif float(np.std(x)) < 1e-12:
            a, b = 1.0, float(np.mean(y - x))
        else:
            a, b = np.polyfit(x, y, deg=1)
            a = float(np.clip(a, 0.5, 1.5))
            b = float(np.clip(b, -2.0, 2.0))
        slopes[j] = a
        intercepts[j] = b
        summary[str(name)] = {"slope": float(a), "intercept": float(b)}
    return slopes, intercepts, summary


def apply_log_calibrators(y_pred: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    pred_log = np.log10(np.maximum(y_pred, 0.0) + EPS)
    calibrated_log = pred_log * slopes.reshape(1, -1) + intercepts.reshape(1, -1)
    calibrated = np.power(10.0, calibrated_log) - EPS
    return np.maximum(calibrated, 0.0)


def per_target_metric_vector(y_true: np.ndarray, y_pred: np.ndarray, eps: float, metric: str) -> np.ndarray:
    err = y_pred - y_true
    abs_err = np.abs(err)
    if metric == "mae":
        return np.mean(abs_err, axis=0)
    if metric == "rmse":
        return np.sqrt(np.mean(err * err, axis=0))
    if metric == "mape":
        return np.mean(abs_err / (np.abs(y_true) + eps), axis=0)
    raise ValueError(f"unsupported metric: {metric}")


def selective_calibration(
    y_true_cal: np.ndarray,
    y_pred_base: np.ndarray,
    y_pred_calibrated: np.ndarray,
    format_names: list[str],
    metric: str,
    eps: float,
) -> tuple[np.ndarray, dict[str, dict[str, float | bool]]]:
    base_scores = per_target_metric_vector(y_true_cal, y_pred_base, eps, metric)
    cal_scores = per_target_metric_vector(y_true_cal, y_pred_calibrated, eps, metric)
    out = y_pred_base.copy()
    summary: dict[str, dict[str, float | bool]] = {}
    for j, name in enumerate(format_names):
        use_cal = bool(cal_scores[j] < base_scores[j])
        if use_cal:
            out[:, j] = y_pred_calibrated[:, j]
        summary[str(name)] = {
            "use_calibrated": use_cal,
            f"base_{metric}": float(base_scores[j]),
            f"calibrated_{metric}": float(cal_scores[j]),
        }
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR / "ml_dataset_ir_errors.npz"))
    ap.add_argument("--out-dir", default=str(DATA_DIR / "eval_in_domain_old_vs_current"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--min-samples-leaf", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=24)
    ap.add_argument("--max-samples", type=float, default=0.25)
    ap.add_argument("--calibration-ratio", type=float, default=0.15)
    ap.add_argument("--calibration-select-metric", default="mape", choices=["mae", "rmse", "mape"])
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]
    ids = data["ids"]
    format_names = data["format_names"].tolist()
    meta = dict(data["meta"].item())

    views = build_sample_level_views(X, Y, ids, format_names, meta)
    sample_ids = views["sample_ids"]
    sample_kernels = views["sample_kernels"]
    X_stats = views["X_stats"]
    Y_sample = views["Y_sample"]
    row_sample_idx = views["row_sample_idx"]
    row_format_idx = views["row_format_idx"]

    tr_samples, te_samples = make_in_domain_split(sample_kernels, float(args.test_ratio), int(args.seed))
    tr_sample_set = set(tr_samples.tolist())
    row_train_mask = np.asarray([idx in tr_sample_set for idx in row_sample_idx.tolist()], dtype=bool)
    row_test_mask = ~row_train_mask

    # Old-style baselines: sample-level multi-output.
    pred_old_stats = fit_predict_multioutput(
        X_stats[tr_samples],
        Y_sample[tr_samples],
        X_stats[te_samples],
        seed=int(args.seed),
        n_estimators=int(args.n_estimators),
        min_samples_leaf=int(args.min_samples_leaf),
        max_depth=int(args.max_depth),
        max_samples=float(args.max_samples),
    )
    # Current model: row-level per-format regression with quant features.
    current_models = train_per_format_models(
        X[row_train_mask],
        Y[row_train_mask],
        row_format_idx[row_train_mask],
        format_names,
        seed=int(args.seed),
        n_estimators=int(args.n_estimators),
        min_samples_leaf=int(args.min_samples_leaf),
        max_depth=int(args.max_depth),
        max_samples=float(args.max_samples),
    )
    test_row_indices = np.where(row_test_mask)[0]
    pred_current_rows = predict_per_format_models(
        current_models,
        X[row_test_mask],
        row_format_idx[row_test_mask],
        format_names,
    )
    pred_current = regroup_row_predictions(
        pred_current_rows,
        test_row_indices,
        te_samples,
        row_sample_idx,
        ids,
        format_names,
    )

    # Calibration: split the train samples again into fit/calibration subsets.
    fit_local, calib_local = make_in_domain_split(
        sample_kernels[tr_samples],
        test_ratio=float(args.calibration_ratio),
        seed=int(args.seed) + 17,
    )
    fit_samples = tr_samples[fit_local]
    calib_samples = tr_samples[calib_local]
    fit_sample_set = set(fit_samples.tolist())
    calib_sample_set = set(calib_samples.tolist())
    row_fit_mask = np.asarray([idx in fit_sample_set for idx in row_sample_idx.tolist()], dtype=bool)
    row_calib_mask = np.asarray([idx in calib_sample_set for idx in row_sample_idx.tolist()], dtype=bool)

    calibrated_models = train_per_format_models(
        X[row_fit_mask],
        Y[row_fit_mask],
        row_format_idx[row_fit_mask],
        format_names,
        seed=int(args.seed),
        n_estimators=int(args.n_estimators),
        min_samples_leaf=int(args.min_samples_leaf),
        max_depth=int(args.max_depth),
        max_samples=float(args.max_samples),
    )
    calib_row_indices = np.where(row_calib_mask)[0]
    pred_calib_rows = predict_per_format_models(
        calibrated_models,
        X[row_calib_mask],
        row_format_idx[row_calib_mask],
        format_names,
    )
    pred_calib = regroup_row_predictions(
        pred_calib_rows,
        calib_row_indices,
        calib_samples,
        row_sample_idx,
        ids,
        format_names,
    )
    calib_true = Y_sample[calib_samples]
    slopes, intercepts, calibration_summary = fit_log_calibrators(calib_true, pred_calib, format_names)

    pred_current_cal_rows = predict_per_format_models(
        calibrated_models,
        X[row_test_mask],
        row_format_idx[row_test_mask],
        format_names,
    )
    pred_current_cal = regroup_row_predictions(
        pred_current_cal_rows,
        test_row_indices,
        te_samples,
        row_sample_idx,
        ids,
        format_names,
    )
    pred_current_cal = apply_log_calibrators(pred_current_cal, slopes, intercepts)

    pred_calib_base_rows = predict_per_format_models(
        calibrated_models,
        X[row_calib_mask],
        row_format_idx[row_calib_mask],
        format_names,
    )
    pred_calib_base = regroup_row_predictions(
        pred_calib_base_rows,
        calib_row_indices,
        calib_samples,
        row_sample_idx,
        ids,
        format_names,
    )
    pred_calib_calibrated = apply_log_calibrators(pred_calib_base, slopes, intercepts)
    _, selective_summary = selective_calibration(
        calib_true,
        pred_calib_base,
        pred_calib_calibrated,
        format_names,
        metric=str(args.calibration_select_metric),
        eps=EPS,
    )
    # Reuse the selected formats on the held-out test set.
    pred_current_selective_test = pred_current.copy()
    for j, name in enumerate(format_names):
        if bool(selective_summary[str(name)]["use_calibrated"]):
            pred_current_selective_test[:, j] = pred_current_cal[:, j]

    y_true = Y_sample[te_samples]

    results = {
        "old_ir2vec_stats_multioutput": metric_dict(y_true, pred_old_stats, EPS),
        "current_per_format_models_quant": metric_dict(y_true, pred_current, EPS),
        "current_per_format_models_quant_calibrated": metric_dict(y_true, pred_current_cal, EPS),
        "current_per_format_models_quant_selective_calibrated": metric_dict(
            y_true, pred_current_selective_test, EPS
        ),
    }

    deltas = {
        "current_minus_old_ir2vec_stats_multioutput": {
            k: float(results["current_per_format_models_quant"][k] - results["old_ir2vec_stats_multioutput"][k])
            for k in results["current_per_format_models_quant"]
        },
        "current_calibrated_minus_current": {
            k: float(
                results["current_per_format_models_quant_calibrated"][k]
                - results["current_per_format_models_quant"][k]
            )
            for k in results["current_per_format_models_quant"]
        },
        "current_selective_minus_current": {
            k: float(
                results["current_per_format_models_quant_selective_calibrated"][k]
                - results["current_per_format_models_quant"][k]
            )
            for k in results["current_per_format_models_quant"]
        },
    }

    summary = {
        "data": str(Path(args.data).resolve()),
        "n_rows": int(X.shape[0]),
        "n_samples": int(sample_ids.shape[0]),
        "n_formats": len(format_names),
        "format_names": format_names,
        "train_samples": int(tr_samples.size),
        "test_samples": int(te_samples.size),
        "train_rows": int(np.sum(row_train_mask)),
        "test_rows": int(np.sum(row_test_mask)),
        "old_ir2vec_stats_x_dim": int(X_stats.shape[1]),
        "current_per_format_x_dim": int(X.shape[1]),
        "seed": int(args.seed),
        "test_ratio": float(args.test_ratio),
        "calibration_ratio": float(args.calibration_ratio),
        "calibration_select_metric": str(args.calibration_select_metric),
        "calibration_samples": int(calib_samples.size),
        "train_config": {
            "n_estimators": int(args.n_estimators),
            "min_samples_leaf": int(args.min_samples_leaf),
            "max_depth": int(args.max_depth),
            "max_samples": float(args.max_samples),
        },
        "calibration": calibration_summary,
        "selective_calibration": selective_summary,
        "results": results,
        "deltas": deltas,
    }

    per_target = []
    per_target.extend(per_target_rows(y_true, pred_old_stats, format_names, "old_ir2vec_stats_multioutput", EPS))
    per_target.extend(
        per_target_rows(y_true, pred_current, format_names, "current_per_format_models_quant", EPS)
    )
    per_target.extend(
        per_target_rows(
            y_true,
            pred_current_cal,
            format_names,
            "current_per_format_models_quant_calibrated",
            EPS,
        )
    )
    per_target.extend(
        per_target_rows(
            y_true,
            pred_current_selective_test,
            format_names,
            "current_per_format_models_quant_selective_calibrated",
            EPS,
        )
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_per_target_csv(out_dir / "per_target.csv", per_target)

    print(f"saved {out_dir / 'summary.json'}")
    print(f"saved {out_dir / 'per_target.csv'}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
