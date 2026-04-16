#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from train_entity_feasibility_model import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    build_xy,
    load_rows,
    subset,
)


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default=str(ROOT / "data" / "generated_entity_label_dataset_accum.csv"),
        help="merged entity label csv",
    )
    ap.add_argument(
        "--model-out",
        default=str(ROOT / "models" / "entity_feasibility_accum_xgb.joblib"),
        help="output joblib model path",
    )
    ap.add_argument(
        "--metrics-out",
        default=str(ROOT / "data" / "entity_feasibility_accum_xgb_metrics.json"),
        help="output metrics json path",
    )
    ap.add_argument(
        "--pred-out",
        default=str(ROOT / "data" / "entity_feasibility_accum_xgb_holdout_preds.csv"),
        help="output holdout predictions csv",
    )
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--reg-lambda", type=float, default=4.0)
    ap.add_argument("--min-child-weight", type=float, default=5.0)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    rows = load_rows(dataset)
    X, y, groups, meta = build_xy(rows)

    splitter = GroupShuffleSplit(n_splits=1, test_size=float(args.test_ratio), random_state=int(args.random_state))
    train_idx, test_idx = next(splitter.split(X, y, groups))
    train_idx = list(train_idx)
    test_idx = list(test_idx)

    X_train = subset(X, train_idx)
    y_train = subset(y, train_idx)
    X_test = subset(X, test_idx)
    y_test = subset(y, test_idx)
    meta_test = subset(meta, test_idx)

    pos = sum(y_train)
    neg = len(y_train) - pos
    scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        learning_rate=float(args.learning_rate),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        reg_lambda=float(args.reg_lambda),
        min_child_weight=float(args.min_child_weight),
        random_state=int(args.random_state),
        n_jobs=1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )
    pipe = Pipeline([("vec", DictVectorizer(sparse=True)), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_score = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset": str(dataset.resolve()),
        "rows_total": len(X),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_apps": len(set(subset(groups, train_idx))),
        "test_apps": len(set(subset(groups, test_idx))),
        "positive_rate_total": sum(y) / len(y),
        "positive_rate_train": sum(y_train) / len(y_train),
        "positive_rate_test": sum(y_test) / len(y_test),
        "model": "XGBClassifier",
        "train_config": {
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "learning_rate": float(args.learning_rate),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "reg_lambda": float(args.reg_lambda),
            "min_child_weight": float(args.min_child_weight),
            "random_state": int(args.random_state),
            "scale_pos_weight": scale_pos_weight,
        },
        "holdout": {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_score),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        },
        "feature_columns": {
            "numeric": NUMERIC_COLUMNS,
            "categorical": CATEGORICAL_COLUMNS,
        },
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "numeric_columns": NUMERIC_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "train_config": metrics["train_config"],
        },
        model_out,
    )

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    pred_out = Path(args.pred_out)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    with pred_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "app",
                "family",
                "entity_id",
                "format",
                "actual_whole_app_rel_err",
                "y_true",
                "y_pred",
                "y_score",
            ],
        )
        writer.writeheader()
        for idx, meta_row in enumerate(meta_test):
            writer.writerow(
                {
                    "app": meta_row["app"],
                    "family": meta_row["family"],
                    "entity_id": meta_row["entity_id"],
                    "format": meta_row["format"],
                    "actual_whole_app_rel_err": meta_row["actual_whole_app_rel_err"],
                    "y_true": meta_row["is_feasible_under_tol"],
                    "y_pred": int(y_pred[idx]),
                    "y_score": float(y_score[idx]),
                }
            )

    print(f"saved {model_out}")
    print(f"saved {metrics_out}")
    print(f"saved {pred_out}")
    print(f"train_rows={len(X_train)} test_rows={len(X_test)}")
    print(f"holdout_accuracy={metrics['holdout']['accuracy']:.6f}")
    print(f"holdout_f1={metrics['holdout']['f1']:.6f}")
    print(f"holdout_roc_auc={metrics['holdout']['roc_auc']:.6f}")


if __name__ == "__main__":
    main()
