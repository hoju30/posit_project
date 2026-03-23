import argparse
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import joblib

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


def parse_format_name_from_id(sample_id: str) -> str:
    return str(sample_id).rsplit("|fmt", 1)[1]


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


def train_format_models(
    X: np.ndarray,
    Y: np.ndarray,
    row_formats: np.ndarray,
    format_names: list[str],
    *,
    n_estimators: int,
    min_samples_leaf: int,
    max_depth: int,
    max_samples: float,
    random_state: int,
    eps: float,
) -> tuple[dict[str, RandomForestRegressor], dict[str, int]]:
    models: dict[str, RandomForestRegressor] = {}
    train_counts: dict[str, int] = {}
    for format_name in format_names:
        mask = row_formats == str(format_name)
        x_fmt = X[mask]
        y_fmt = Y[mask]
        y_log = np.log10(np.maximum(y_fmt, 0.0) + eps)
        reg = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_features="sqrt",
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )
        reg.fit(x_fmt, y_log)
        models[str(format_name)] = reg
        train_counts[str(format_name)] = int(mask.sum())
    return models, train_counts


def predict_with_models(
    models: dict[str, RandomForestRegressor],
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


def regroup_predictions(
    pred_rows: np.ndarray,
    row_indices: np.ndarray,
    target_samples: np.ndarray,
    row_sample_idx: np.ndarray,
    row_format_idx: np.ndarray,
    n_formats: int,
) -> np.ndarray:
    pred = np.full((target_samples.size, n_formats), np.nan, dtype=np.float32)
    sample_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(target_samples.tolist())}
    for row_idx, val in zip(row_indices.tolist(), pred_rows.tolist()):
        global_sample_idx = int(row_sample_idx[row_idx])
        local_sample_idx = sample_to_local[global_sample_idx]
        fmt_idx = int(row_format_idx[row_idx])
        pred[local_sample_idx, fmt_idx] = float(val)
    if np.isnan(pred).any():
        raise RuntimeError("calibration regroup failed: incomplete per-format matrix")
    return pred


def fit_log_calibrators(y_true: np.ndarray, y_pred: np.ndarray, format_names: list[str], eps: float):
    slopes = np.ones((len(format_names),), dtype=np.float64)
    intercepts = np.zeros((len(format_names),), dtype=np.float64)
    summary: dict[str, dict[str, float]] = {}
    for j, name in enumerate(format_names):
        x = np.log10(np.maximum(y_pred[:, j], 0.0) + eps)
        y = np.log10(np.maximum(y_true[:, j], 0.0) + eps)
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


def apply_log_calibrators(y_pred: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray, eps: float) -> np.ndarray:
    pred_log = np.log10(np.maximum(y_pred, 0.0) + eps)
    calibrated_log = pred_log * slopes.reshape(1, -1) + intercepts.reshape(1, -1)
    calibrated = np.power(10.0, calibrated_log) - eps
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


def selective_calibration_summary(
    y_true_cal: np.ndarray,
    y_pred_base: np.ndarray,
    y_pred_calibrated: np.ndarray,
    format_names: list[str],
    metric: str,
    eps: float,
) -> dict[str, dict[str, float | bool]]:
    base_scores = per_target_metric_vector(y_true_cal, y_pred_base, eps, metric)
    cal_scores = per_target_metric_vector(y_true_cal, y_pred_calibrated, eps, metric)
    out: dict[str, dict[str, float | bool]] = {}
    for j, name in enumerate(format_names):
        out[str(name)] = {
            "use_calibrated": bool(cal_scores[j] < base_scores[j]),
            f"base_{metric}": float(base_scores[j]),
            f"calibrated_{metric}": float(cal_scores[j]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-estimators", type=int, default=120)
    ap.add_argument("--min-samples-leaf", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=20)
    ap.add_argument("--max-samples", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--calibration-ratio", type=float, default=0.15)
    ap.add_argument("--calibration-select-metric", default="mape", choices=["mae", "rmse", "mape"])
    args = ap.parse_args()

    data = np.load(DATA_DIR / "ml_dataset_ir_errors.npz", allow_pickle=True)
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

    # Build calibration plan on a held-out subset of the training data, then retrain final models on all rows.
    tr_samples, cal_samples = make_in_domain_split(
        sample_kernels,
        test_ratio=float(args.calibration_ratio),
        seed=int(args.random_state) + 17,
    )
    tr_sample_set = set(tr_samples.tolist())
    cal_sample_set = set(cal_samples.tolist())
    row_train_mask = np.asarray([idx in tr_sample_set for idx in row_sample_idx.tolist()], dtype=bool)
    row_cal_mask = np.asarray([idx in cal_sample_set for idx in row_sample_idx.tolist()], dtype=bool)

    cal_models, _ = train_format_models(
        X[row_train_mask],
        Y[row_train_mask],
        row_formats[row_train_mask],
        format_names,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
        random_state=args.random_state,
        eps=eps,
    )
    pred_cal_rows = predict_with_models(
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

    models, train_counts = train_format_models(
        X,
        Y,
        row_formats,
        format_names,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
        random_state=args.random_state,
        eps=eps,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "format_names": format_names,
            "meta": dict(meta.item()) if meta is not None else None,
            "y_transform": {"type": "log10", "eps": eps},
            "model_kind": "per_format_models",
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
                "n_estimators": args.n_estimators,
                "min_samples_leaf": args.min_samples_leaf,
                "max_depth": args.max_depth,
                "max_samples": args.max_samples,
                "random_state": args.random_state,
            },
        },
        MODEL_DIR / "ir2vec_error_predictor.joblib",
        compress=3,
    )

if __name__ == "__main__":
    main()
