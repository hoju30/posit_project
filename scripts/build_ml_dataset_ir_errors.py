import numpy as np
import pandas as pd
from pathlib import Path

from error_feature_utils import (
    CURRENT_ES_DERIVED_FIELDS,
    DEFAULT_EPS,
    DERIVED_STAT_FIELDS,
    FORMAT_DEP_FIELDS,
    RAW_STAT_FIELDS,
    current_es_excess_features,
    derived_stats_features,
    prefixed_current_es_names,
    prefixed_derived_stat_names,
    prefixed_quant_feat_names,
    prefixed_raw_stat_names,
)

# class 空間限定 posit(8/16/32, es=0..2 或合法範圍內)
def generate_formats():
    fmts = []
    for n in (8, 16, 32):
        max_es = min(2, n - 2)
        for es in range(max_es + 1):
            fmts.append((n, es))
    return fmts

POSIT_FORMATS = generate_formats()
FORMAT_NAMES = [f"posit_{n}_{es}" for (n, es) in POSIT_FORMATS]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IR_DIR = ROOT / "ir"

KERNEL_DATASETS = {
    "dot":         "dot_dataset.csv",
    "sum":         "sum_dataset.csv",
    "axpy_sum":    "axpy_sum_dataset.csv",
    "relu_sum":    "relu_sum_dataset.csv",
    "l1_norm":     "l1_norm_dataset.csv",
    "sum_squares": "sum_squares_dataset.csv",
    "axpby_sum":   "axpby_sum_dataset.csv",
    "matvec":      "matvec_dataset.csv",
    "prefix_sum":  "prefix_sum_dataset.csv",
}

GROUP_KEYS = {
    "matvec": ["rows", "cols", "dist_type"],
}

# 每個 setting 最多取多少 sample_id 轉成訓練點。
# 提高到 500，讓既有大多數 kernel 的 500 筆生成資料都能完整進入訓練集。
MAX_SAMPLES_PER_SETTING = 500
EPS = DEFAULT_EPS
STATS_DIM_PER_VEC = len(RAW_STAT_FIELDS)  # per x or y
DERIVED_DIM_PER_VEC = len(DERIVED_STAT_FIELDS)
CURRENT_ES_DIM_PER_VEC = len(CURRENT_ES_DERIVED_FIELDS)

def expand_group_to_samples(df: pd.DataFrame, kernel_id: str):
    """
    以 group key 分組後，針對每個 group 下的 sample_id 做多筆樣本。
    每個樣本保留所有 posit 格式的 rel_err，後續再展開成 per-format 訓練點。
    """
    gkeys = GROUP_KEYS.get(kernel_id, ["vec_len", "dist_type"])
    rows = []

    grouped = df.groupby(gkeys)
    for gvals, gdf in grouped:
        if not isinstance(gvals, tuple):
            gvals = (gvals,)
        key_map = dict(zip(gkeys, gvals))
        vec_len = key_map.get("vec_len", key_map.get("cols", None))
        dist_type = key_map.get("dist_type", None)
        rows_val = key_map.get("rows", None)
        cols_val = key_map.get("cols", None)

        sample_ids = gdf["sample_id"].unique()
        sample_ids = sample_ids[:MAX_SAMPLES_PER_SETTING]

        for sid in sample_ids:
            sub_df = gdf[gdf["sample_id"] == sid]

            targets: dict[str, float] = {}
            format_dependent: dict[str, dict[str, float]] = {}
            for n, es in POSIT_FORMATS:
                sub = sub_df[(sub_df["posit_n"] == n) & (sub_df["posit_es"] == es)]
                name = f"posit_{n}_{es}"
                targets[name] = np.nan if sub.empty else float(sub["rel_err"].mean())
                qvals: dict[str, float] = {}
                for prefix in ("x", "y"):
                    for field in FORMAT_DEP_FIELDS:
                        col = f"{prefix}_{field}"
                        qvals[col] = float(sub[col].mean()) if (not sub.empty and col in sub.columns) else 0.0
                format_dependent[name] = qvals

            has_stats = all(
                f"{prefix}_{name}" in sub_df.columns
                for prefix in ("x", "y")
                for name in RAW_STAT_FIELDS
            )
            stat_values: dict[str, float] = {}
            for prefix in ("x", "y"):
                for name in RAW_STAT_FIELDS:
                    col = f"{prefix}_{name}"
                    stat_values[col] = float(sub_df[col].mean()) if col in sub_df else 0.0

            rows.append({
                "kernel_id": kernel_id,
                "vec_len": vec_len,
                "dist_type": dist_type,
                "rows": rows_val,
                "cols": cols_val,
                "sample_id": sid,
                "targets": targets,
                "format_dependent": format_dependent,
                "has_stats": has_stats,
                "alpha": float(sub_df["alpha"].mean()) if "alpha" in sub_df else 0.0,
                "beta": float(sub_df["beta"].mean()) if "beta" in sub_df else 0.0,
                **stat_values,
            })
    return rows


def main():
    feats = np.load(DATA_DIR / "ir2vec_features.npz", allow_pickle=True)
    X_all = feats["X"]
    ids_all = feats["ids"]
    id2vec = {str(kid): X_all[i] for i, kid in enumerate(ids_all)}

    def find_kernel_ll(kernel_id: str) -> Path | None:
        p = IR_DIR / f"{kernel_id}.ll"
        if p.exists():
            return p
        return None

    id2cfg: dict[str, np.ndarray] = {}
    cfg_dim = 0
    for kid in id2vec.keys():
        id2cfg[kid] = np.zeros((cfg_dim,), dtype=np.float32)

    records = []
    for kernel_id, csv_name in KERNEL_DATASETS.items():
        df = pd.read_csv(DATA_DIR / csv_name)
        records.extend(expand_group_to_samples(df, kernel_id))

    # 取得縮放因子
    vec_vals = [r["vec_len"] for r in records if r["vec_len"] is not None]
    row_vals = [r["rows"] for r in records if r["rows"] is not None]
    col_vals = [r["cols"] for r in records if r["cols"] is not None]

    vec_max = max(vec_vals or [1])
    rows_max = max(row_vals or [1])
    cols_max = max(col_vals or [1])

    # 展開資料集：
    # X = ir2vec + shared features(vec_len/rows/cols/alpha/beta + x/y stats)
    #   + format features(x/y quant features)
    # Y = 該 format 的單一 rel_err
    X_list, Y_list, ids_list = [], [], []
    for rec in records:
        kid = rec["kernel_id"]
        if kid not in id2vec:
            print(f"[warn] skip {kid} (no IR2Vec feature)")
            continue

        base_vec = id2vec[kid]
        vec_len = float(rec["vec_len"] or 0.0)
        rows_v = float(rec["rows"] or 0.0)
        cols_v = float(rec["cols"] or 0.0)

        extra = []
        extra.append(vec_len / vec_max if vec_max else 0.0)
        extra.append(rows_v / rows_max if rows_max else 0.0)
        extra.append(cols_v / cols_max if cols_max else 0.0)
        # axpy/axpby 類 kernel 的係數（其他 kernel 會是 0）
        extra.append(float(rec.get("alpha", 0.0)))
        extra.append(float(rec.get("beta", 0.0)))

        if not rec.get("has_stats", False):
            stats_feats = [0.0] * (2 * STATS_DIM_PER_VEC + 2 * DERIVED_DIM_PER_VEC)
        else:
            x_stats_map = {name: float(rec.get(f"x_{name}", 0.0)) for name in RAW_STAT_FIELDS}
            y_stats_map = {name: float(rec.get(f"y_{name}", 0.0)) for name in RAW_STAT_FIELDS}
            x_feats = [float(x_stats_map[name]) for name in RAW_STAT_FIELDS]
            y_feats = [float(y_stats_map[name]) for name in RAW_STAT_FIELDS]
            x_derived = derived_stats_features(x_stats_map, eps=EPS)
            y_derived = derived_stats_features(y_stats_map, eps=EPS)
            stats_feats = x_feats + y_feats + x_derived + y_derived

        cfg_feats = id2cfg.get(kid, np.zeros((cfg_dim,), dtype=np.float32))
        shared_feat = np.concatenate([base_vec, np.array(extra + stats_feats, dtype=np.float32), cfg_feats])

        for n, es in POSIT_FORMATS:
            name = f"posit_{n}_{es}"
            target = float(rec["targets"].get(name, np.nan))
            if np.isnan(target):
                continue
            quant_map = rec.get("format_dependent", {}).get(name, {})
            quant_feats = [float(quant_map.get(f"x_{field}", 0.0)) for field in FORMAT_DEP_FIELDS] + [
                float(quant_map.get(f"y_{field}", 0.0)) for field in FORMAT_DEP_FIELDS
            ]
            x_stats_map = {field: float(rec.get(f"x_{field}", 0.0)) for field in RAW_STAT_FIELDS}
            y_stats_map = {field: float(rec.get(f"y_{field}", 0.0)) for field in RAW_STAT_FIELDS}
            current_es_feats = current_es_excess_features(x_stats_map, es=es, eps=EPS) + current_es_excess_features(
                y_stats_map, es=es, eps=EPS
            )
            format_feats = np.array(current_es_feats + quant_feats, dtype=np.float32)
            feat = np.concatenate([shared_feat, format_feats])

            X_list.append(feat)
            Y_list.append(target)
            ids_list.append(
                f"{kid}|len{rec['vec_len']}|dist{rec['dist_type']}|sid{rec['sample_id']}|fmt{name}"
            )

    if not X_list:
        raise RuntimeError("No data to build regression dataset.")

    X_out = np.stack(X_list, axis=0)
    Y_out = np.asarray(Y_list, dtype=np.float32)
    stats_feat_names = (
        prefixed_raw_stat_names("x")
        + prefixed_raw_stat_names("y")
        + prefixed_derived_stat_names("x")
        + prefixed_derived_stat_names("y")
    )
    current_es_feat_names = prefixed_current_es_names("x") + prefixed_current_es_names("y")
    quant_feat_names = prefixed_quant_feat_names("x") + prefixed_quant_feat_names("y")
    format_feat_names = current_es_feat_names + quant_feat_names
    shared_feat_dim = 5 + 2 * STATS_DIM_PER_VEC + 2 * DERIVED_DIM_PER_VEC + cfg_dim
    format_feat_dim = len(format_feat_names)

    np.savez(
        DATA_DIR / "ml_dataset_ir_errors.npz",
        X=X_out,
        Y=Y_out,
        ids=np.array(ids_list),
        format_names=np.array(FORMAT_NAMES),
        meta=np.array({
            "ir_dim": X_all.shape[1],
            "vec_scale": vec_max,
            "rows_scale": rows_max,
            "cols_scale": cols_max,
            "cfg_dim": cfg_dim,
            "cfg_feat_names": [],
            "shared_feat_dim": shared_feat_dim,
            "format_feat_dim": format_feat_dim,
            "stats_dim": len(stats_feat_names),
            "current_es_dim": len(current_es_feat_names),
            "quant_dim": len(quant_feat_names),
            "extra_dim": shared_feat_dim + format_feat_dim,
            "format_feat_names": format_feat_names,
            "format_names": FORMAT_NAMES,
            "format_specs": POSIT_FORMATS,
            "vec_lens": sorted(set(vec_vals)),
            "rows_vals": sorted(set(row_vals)),
            "cols_vals": sorted(set(col_vals)),
            "stats_feat_names": stats_feat_names,
            "current_es_feat_names": current_es_feat_names,
            "quant_feat_names": quant_feat_names,
            "eps": EPS,
        }, dtype=object),
    )
    print("saved ml_dataset_ir_errors.npz")
    print("X shape:", X_out.shape)
    print("Y shape:", Y_out.shape)
    print("formats:", FORMAT_NAMES)


if __name__ == "__main__":
    main()
