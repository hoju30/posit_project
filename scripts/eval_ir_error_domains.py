#!/usr/bin/env python3
"""
Evaluate IR error regression model on:
  1) in-domain split (stratified by kernel)
  2) out-of-domain split (hold out selected kernel(s))

Outputs:
  - summary.json
  - <feature_set>_in_domain_per_target.csv
  - <feature_set>_in_domain_pred_vs_actual.csv
  - <feature_set>_ood_<kernel>_per_target.csv
  - <feature_set>_ood_<kernel>_pred_vs_actual.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def parse_kernel_ids(ids: np.ndarray) -> np.ndarray:
    return np.asarray([str(s).split("|", 1)[0] for s in ids], dtype=object)


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


def feature_slice_map(meta: dict[str, object]) -> dict[str, slice]:
    ir_dim = int(meta.get("ir_dim", 300))
    cfg_dim = int(meta.get("cfg_dim", 0) or 0)
    extra_dim = int(meta.get("extra_dim", 0) or 0)
    stats_dim = len(meta.get("stats_feat_names", []))
    scalar_dim = extra_dim - stats_dim - cfg_dim
    if scalar_dim < 0:
        raise ValueError("invalid feature metadata: scalar_dim < 0")

    ir_end = ir_dim
    scalar_end = ir_end + scalar_dim
    stats_end = scalar_end + stats_dim
    cfg_end = stats_end + cfg_dim
    return {
        "ir2vec_only": slice(0, ir_end),
        "ir2vec_stats": slice(0, stats_end),
        "ir2vec_stats_cfg": slice(0, cfg_end),
    }


def select_features(X: np.ndarray, feat_name: str, meta: dict[str, object]) -> np.ndarray:
    smap = feature_slice_map(meta)
    if feat_name not in smap:
        raise ValueError(f"unsupported feature set: {feat_name}")
    return X[:, smap[feat_name]]


def make_model(seed: int, n_estimators: int, min_samples_leaf: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


def fit_predict(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    eps: float,
    n_estimators: int,
    min_samples_leaf: int,
) -> np.ndarray:
    model = make_model(seed=seed, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
    Y_train_log = np.log10(np.maximum(Y_train, 0.0) + eps)
    model.fit(X_train, Y_train_log)
    pred_log = model.predict(X_test)
    pred = np.power(10.0, pred_log) - eps
    pred = np.maximum(pred, 0.0)
    return pred


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err * err)))
    mape = float(np.mean(abs_err / (np.abs(y_true) + eps)))

    y_true_best_idx = np.argmin(y_true, axis=1)
    y_pred_best_idx = np.argmin(y_pred, axis=1)
    pick_acc = float(np.mean(y_true_best_idx == y_pred_best_idx))

    row_ids = np.arange(y_true.shape[0])
    oracle = y_true[row_ids, y_true_best_idx]
    chosen_actual = y_true[row_ids, y_pred_best_idx]
    regret = chosen_actual - oracle
    mean_regret = float(np.mean(regret))
    median_regret = float(np.median(regret))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "pick_acc": pick_acc,
        "mean_regret": mean_regret,
        "median_regret": median_regret,
    }


def per_target_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    eps: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for j, t in enumerate(target_names):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        err = yp - yt
        abs_err = np.abs(err)
        rows.append(
            {
                "target": t,
                "mae": float(np.mean(abs_err)),
                "rmse": float(np.sqrt(np.mean(err * err))),
                "mape": float(np.mean(abs_err / (np.abs(yt) + eps))),
            }
        )
    return rows


def write_per_target_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["target", "mae", "rmse", "mape"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_pred_vs_actual_csv(
    path: Path,
    ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "target", "actual", "pred"])
        for i in range(y_true.shape[0]):
            sid = str(ids[i])
            for j, t in enumerate(target_names):
                w.writerow([sid, t, float(y_true[i, j]), float(y_pred[i, j])])


def make_in_domain_split(kernels: np.ndarray, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for k in sorted(set(kernels.tolist())):
        idx = np.where(kernels == k)[0]
        if idx.size < 2:
            train_parts.append(idx)
            continue
        n_test = int(round(idx.size * test_ratio))
        n_test = max(1, min(n_test, idx.size - 1))
        chosen = rng.choice(idx, size=n_test, replace=False)
        mask = np.ones(idx.size, dtype=bool)
        chosen_set = set(chosen.tolist())
        for i, v in enumerate(idx.tolist()):
            if v in chosen_set:
                mask[i] = False
        train_parts.append(idx[mask])
        test_parts.append(np.asarray(chosen, dtype=np.int64))

    train_idx = np.concatenate(train_parts) if train_parts else np.zeros((0,), dtype=np.int64)
    test_idx = np.concatenate(test_parts) if test_parts else np.zeros((0,), dtype=np.int64)
    return train_idx, test_idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR / "ml_dataset_ir_errors.npz"))
    ap.add_argument("--out-dir", default=str(DATA_DIR / "eval_domains"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--eps", type=float, default=1e-30)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--min-samples-leaf", type=int, default=2)
    ap.add_argument(
        "--ood-kernels",
        default="dot",
        help="comma-separated kernels or 'all' (default: dot)",
    )
    ap.add_argument(
        "--skip-ood",
        action="store_true",
        help="only run in-domain evaluation",
    )
    ap.add_argument(
        "--feature-sets",
        default="all",
        help="comma-separated: ir2vec_only,ir2vec_stats,ir2vec_stats_cfg or 'all'",
    )
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]
    ids = data["ids"]
    target_names = data["target_names"].tolist()
    meta = dict(data["meta"].item()) if "meta" in data else {}
    kernels = parse_kernel_ids(ids)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "data": str(Path(args.data).resolve()),
        "n_samples": int(X.shape[0]),
        "x_dim": int(X.shape[1]),
        "y_dim": int(Y.shape[1]),
        "targets": target_names,
        "feature_meta": meta,
        "kernel_counts": {k: int(np.sum(kernels == k)) for k in sorted(set(kernels.tolist()))},
        "results": {},
    }

    if args.feature_sets.strip().lower() == "all":
        feature_sets = ["ir2vec_only", "ir2vec_stats", "ir2vec_stats_cfg"]
    else:
        feature_sets = [s.strip() for s in args.feature_sets.split(",") if s.strip()]

    tr_idx, te_idx = make_in_domain_split(kernels, test_ratio=float(args.test_ratio), seed=int(args.seed))

    all_kernels = sorted(set(kernels.tolist()))
    if args.skip_ood:
        holdouts = []
    elif args.ood_kernels.strip().lower() == "all":
        holdouts = all_kernels
    else:
        holdouts = [s.strip() for s in args.ood_kernels.split(",") if s.strip()]

    for feat_name in feature_sets:
        X_sel = select_features(X, feat_name, meta)
        feat_summary: dict[str, object] = {
            "x_dim": int(X_sel.shape[1]),
            "in_domain": {},
            "ood": {},
        }

        y_pred_in = fit_predict(
            X_sel[tr_idx],
            Y[tr_idx],
            X_sel[te_idx],
            seed=int(args.seed),
            eps=float(args.eps),
            n_estimators=int(args.n_estimators),
            min_samples_leaf=int(args.min_samples_leaf),
        )
        in_overall = metric_dict(Y[te_idx], y_pred_in, eps=float(args.eps))
        in_per_target = per_target_rows(Y[te_idx], y_pred_in, target_names, eps=float(args.eps))
        write_per_target_csv(out_dir / f"{feat_name}_in_domain_per_target.csv", in_per_target)
        write_pred_vs_actual_csv(
            out_dir / f"{feat_name}_in_domain_pred_vs_actual.csv",
            ids[te_idx],
            Y[te_idx],
            y_pred_in,
            target_names,
        )
        feat_summary["in_domain"] = {
            "train_size": int(tr_idx.size),
            "test_size": int(te_idx.size),
            "metrics": in_overall,
        }

        for hk in holdouts:
            te = np.where(kernels == hk)[0]
            tr = np.where(kernels != hk)[0]
            if te.size == 0 or tr.size == 0:
                continue
            y_pred_ood = fit_predict(
                X_sel[tr],
                Y[tr],
                X_sel[te],
                seed=int(args.seed),
                eps=float(args.eps),
                n_estimators=int(args.n_estimators),
                min_samples_leaf=int(args.min_samples_leaf),
            )
            ood_overall = metric_dict(Y[te], y_pred_ood, eps=float(args.eps))
            ood_per_target = per_target_rows(Y[te], y_pred_ood, target_names, eps=float(args.eps))
            write_per_target_csv(out_dir / f"{feat_name}_ood_{hk}_per_target.csv", ood_per_target)
            write_pred_vs_actual_csv(
                out_dir / f"{feat_name}_ood_{hk}_pred_vs_actual.csv",
                ids[te],
                Y[te],
                y_pred_ood,
                target_names,
            )
            feat_summary["ood"][hk] = {
                "train_size": int(tr.size),
                "test_size": int(te.size),
                "metrics": ood_overall,
            }

        summary["results"][feat_name] = feat_summary

    (out_dir / "summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Saved:", out_dir / "summary.json")
    for feat_name in feature_sets:
        feat_summary = summary["results"][feat_name]
        print(f"{feat_name} in-domain:", feat_summary["in_domain"])
        print(f"{feat_name} OOD kernels:", sorted((feat_summary["ood"] or {}).keys()))


if __name__ == "__main__":
    main()
