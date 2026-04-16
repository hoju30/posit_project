#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from train_entity_feasibility_model import build_xy, load_rows, subset


ROOT = Path(__file__).resolve().parent.parent

NUMERIC_GROUPS = {
    "use_count": "local",
    "is_loop_carried": "local",
    "is_output_related": "local",
    "fanout": "graph",
    "reduction_depth": "graph",
    "distance_to_output": "graph",
    "same_loop_neighbor_count": "graph",
    "num_arith_users": "graph",
    "format_n": "format",
    "format_es": "format",
}

CATEGORICAL_GROUPS = {
    "family": "context",
    "role": "role",
    "def_opcode": "opcode",
    "function": "context",
    "bb": "context",
    "format": "format",
}

ABLATIONS: dict[str, set[str]] = {
    "all": set(),
    "no_role": {"role"},
    "no_graph": {"graph"},
    "no_format": {"format"},
    "no_context": {"context"},
    "no_local": {"local"},
    "no_opcode": {"opcode"},
}


def feature_lists(excluded_groups: set[str]) -> tuple[list[str], list[str]]:
    numeric = [name for name, group in NUMERIC_GROUPS.items() if group not in excluded_groups]
    categorical = [name for name, group in CATEGORICAL_GROUPS.items() if group not in excluded_groups]
    return numeric, categorical


def build_xy_subset(
    rows: list[dict[str, str]], numeric_cols: list[str], categorical_cols: list[str]
) -> tuple[list[dict[str, object]], list[int], list[str]]:
    X, y, groups, _meta = build_xy(rows)
    filtered_X: list[dict[str, object]] = []
    for row in X:
        feat = {k: row[k] for k in numeric_cols + categorical_cols}
        filtered_X.append(feat)
    return filtered_X, y, groups


def train_rf(X_train: list[dict[str, object]], y_train: list[int]) -> Pipeline:
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        random_state=0,
        n_jobs=1,
        class_weight="balanced_subsample",
    )
    pipe = Pipeline([("vec", DictVectorizer(sparse=True)), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgb(X_train: list[dict[str, object]], y_train: list[int]) -> Pipeline:
    pos = sum(y_train)
    neg = len(y_train) - pos
    scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=4.0,
        min_child_weight=5.0,
        random_state=0,
        n_jobs=1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )
    pipe = Pipeline([("vec", DictVectorizer(sparse=True)), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def eval_pipe(pipe: Pipeline, X_test: list[dict[str, object]], y_test: list[int]) -> dict[str, float]:
    y_pred = pipe.predict(X_test)
    y_score = pipe.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
    }


def importance_rows(pipe: Pipeline) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    vec: DictVectorizer = pipe.named_steps["vec"]
    clf = pipe.named_steps["clf"]
    names = vec.get_feature_names_out()
    importances = clf.feature_importances_

    by_group: dict[str, float] = defaultdict(float)
    rows: list[dict[str, object]] = []

    group_lookup = {}
    group_lookup.update(NUMERIC_GROUPS)
    group_lookup.update(CATEGORICAL_GROUPS)

    for name, importance in zip(names, importances):
        base_feature = name.split("=", 1)[0]
        feature_group = group_lookup.get(base_feature, "unknown")
        value = float(importance)
        rows.append(
            {
                "vectorized_feature": name,
                "base_feature": base_feature,
                "feature_group": feature_group,
                "importance": value,
            }
        )
        by_group[feature_group] += value

    rows.sort(key=lambda r: r["importance"], reverse=True)
    group_rows = [
        {"feature_group": group, "importance": value}
        for group, value in sorted(by_group.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return rows, group_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default=str(ROOT / "data" / "generated_entity_label_dataset_all.csv"),
        help="merged unified entity dataset",
    )
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "data" / "eval_entity_feasibility_unified_features"),
        help="output directory",
    )
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=0)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(dataset)
    base_numeric, base_categorical = feature_lists(set())
    X_all, y_all, groups = build_xy_subset(rows, base_numeric, base_categorical)

    splitter = GroupShuffleSplit(n_splits=1, test_size=float(args.test_ratio), random_state=int(args.random_state))
    train_idx, test_idx = next(splitter.split(X_all, y_all, groups))
    train_idx = list(train_idx)
    test_idx = list(test_idx)

    ablation_rows: list[dict[str, object]] = []

    for name, excluded in ABLATIONS.items():
        numeric_cols, categorical_cols = feature_lists(excluded)
        X, y, _groups = build_xy_subset(rows, numeric_cols, categorical_cols)
        X_train = subset(X, train_idx)
        y_train = subset(y, train_idx)
        X_test = subset(X, test_idx)
        y_test = subset(y, test_idx)

        rf_pipe = train_rf(X_train, y_train)
        rf_metrics = eval_pipe(rf_pipe, X_test, y_test)
        ablation_rows.append(
            {
                "model": "rf",
                "ablation": name,
                "excluded_groups": ",".join(sorted(excluded)),
                "numeric_columns": ",".join(numeric_cols),
                "categorical_columns": ",".join(categorical_cols),
                **rf_metrics,
            }
        )

        xgb_pipe = train_xgb(X_train, y_train)
        xgb_metrics = eval_pipe(xgb_pipe, X_test, y_test)
        ablation_rows.append(
            {
                "model": "xgb",
                "ablation": name,
                "excluded_groups": ",".join(sorted(excluded)),
                "numeric_columns": ",".join(numeric_cols),
                "categorical_columns": ",".join(categorical_cols),
                **xgb_metrics,
            }
        )

        if name == "all":
            rf_feature_rows, rf_group_rows = importance_rows(rf_pipe)
            xgb_feature_rows, xgb_group_rows = importance_rows(xgb_pipe)
            write_csv(
                out_dir / "rf_feature_importance.csv",
                rf_feature_rows,
                ["vectorized_feature", "base_feature", "feature_group", "importance"],
            )
            write_csv(
                out_dir / "rf_group_importance.csv",
                rf_group_rows,
                ["feature_group", "importance"],
            )
            write_csv(
                out_dir / "xgb_feature_importance.csv",
                xgb_feature_rows,
                ["vectorized_feature", "base_feature", "feature_group", "importance"],
            )
            write_csv(
                out_dir / "xgb_group_importance.csv",
                xgb_group_rows,
                ["feature_group", "importance"],
            )

    write_csv(
        out_dir / "ablation.csv",
        ablation_rows,
        [
            "model",
            "ablation",
            "excluded_groups",
            "accuracy",
            "f1",
            "roc_auc",
            "numeric_columns",
            "categorical_columns",
        ],
    )

    summary = {
        "dataset": str(dataset.resolve()),
        "rows_total": len(X_all),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_apps": len(set(subset(groups, train_idx))),
        "test_apps": len(set(subset(groups, test_idx))),
        "ablations": {name: sorted(groups) for name, groups in ABLATIONS.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"saved {out_dir / 'summary.json'}")
    print(f"saved {out_dir / 'ablation.csv'}")
    print(f"saved {out_dir / 'rf_feature_importance.csv'}")
    print(f"saved {out_dir / 'rf_group_importance.csv'}")
    print(f"saved {out_dir / 'xgb_feature_importance.csv'}")
    print(f"saved {out_dir / 'xgb_group_importance.csv'}")


if __name__ == "__main__":
    main()
