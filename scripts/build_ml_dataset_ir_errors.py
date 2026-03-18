# build_ml_dataset_ir_errors.py
# 產生「預測各格式相對誤差」用的回歸資料集
import numpy as np
import pandas as pd
from pathlib import Path

from split_loops_and_analyze_cfg import cfg_feature_vector_from_ll_path, CFG_FEAT_NAMES

# class 空間限定 posit(8/16/32, es=0..2 或合法範圍內)
def generate_formats():
    fmts = []
    for n in (8, 16, 32):
        max_es = min(2, n - 2)
        for es in range(max_es + 1):
            fmts.append((n, es))
    return fmts

POSIT_FORMATS = generate_formats()
TARGET_NAMES = [f"posit_{n}_{es}" for (n, es) in POSIT_FORMATS] + ["fp32"]

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
EPS = 1e-30
ES_LIST = [0, 1, 2]
USEED_LOG10 = {es: np.log10(2.0 ** (2 ** es)) for es in ES_LIST}
STATS_DIM_PER_VEC = 12  # per x or y

def expand_group_to_samples(df: pd.DataFrame, kernel_id: str):
    """
    以 group key 分組後，針對每個 group 下的 sample_id 做多筆樣本。
    每個樣本產生一個 target 向量（各 posit 的 rel_err、fp32；皆為相對 fp64 的 rel_err）。
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

            targets = []
            for n, es in POSIT_FORMATS:
                sub = sub_df[(sub_df["posit_n"] == n) & (sub_df["posit_es"] == es)]
                if sub.empty:
                    targets.append(np.nan)
                else:
                    targets.append(sub["rel_err"].mean())

            fp32_err = sub_df["fp32_rel_err"].mean() if "fp32_rel_err" in sub_df else np.nan
            targets.append(fp32_err)

            # dot 的輸入統計（若欄位存在）；其他 kernel 會是 0
            has_stats = all(col in sub_df.columns for col in ("x_p1", "x_p50", "x_p99", "y_p1", "y_p50", "y_p99"))
            x_mean = float(sub_df["x_mean"].mean()) if "x_mean" in sub_df else 0.0
            x_std = float(sub_df["x_std"].mean()) if "x_std" in sub_df else 0.0
            x_p1 = float(sub_df["x_p1"].mean()) if "x_p1" in sub_df else 0.0
            x_p50 = float(sub_df["x_p50"].mean()) if "x_p50" in sub_df else 0.0
            x_p99 = float(sub_df["x_p99"].mean()) if "x_p99" in sub_df else 0.0
            y_mean = float(sub_df["y_mean"].mean()) if "y_mean" in sub_df else 0.0
            y_std = float(sub_df["y_std"].mean()) if "y_std" in sub_df else 0.0
            y_p1 = float(sub_df["y_p1"].mean()) if "y_p1" in sub_df else 0.0
            y_p50 = float(sub_df["y_p50"].mean()) if "y_p50" in sub_df else 0.0
            y_p99 = float(sub_df["y_p99"].mean()) if "y_p99" in sub_df else 0.0

            rows.append({
                "kernel_id": kernel_id,
                "vec_len": vec_len,
                "dist_type": dist_type,
                "rows": rows_val,
                "cols": cols_val,
                "sample_id": sid,
                "targets": targets,
                "x_mean": x_mean, "x_std": x_std, "x_p1": x_p1, "x_p50": x_p50, "x_p99": x_p99,
                "y_mean": y_mean, "y_std": y_std, "y_p1": y_p1, "y_p50": y_p50, "y_p99": y_p99,
                "has_stats": has_stats,
                "alpha": float(sub_df["alpha"].mean()) if "alpha" in sub_df else 0.0,
                "beta": float(sub_df["beta"].mean()) if "beta" in sub_df else 0.0,
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

    # Per-kernel CFG summary features (segment-level features for the kernel IR)
    id2cfg: dict[str, np.ndarray] = {}
    cfg_dim = len(CFG_FEAT_NAMES)
    for kid in id2vec.keys():
        ll = find_kernel_ll(kid)
        if ll is None:
            print(f"[warn] CFG features: missing IR .ll for {kid}, fill zeros")
            id2cfg[kid] = np.zeros((cfg_dim,), dtype=np.float32)
            continue
        try:
            v, names = cfg_feature_vector_from_ll_path(ll, scc_mode="auto")
            if len(v) != cfg_dim:
                raise ValueError("cfg dim mismatch")
            id2cfg[kid] = v
        except Exception as e:
            print(f"[warn] CFG features: failed on {kid}: {e}; fill zeros")
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
    # X = ir2vec 特徵 + 附加特徵(vec_len/rows/cols) + dot 統計（由 std/min/max 推導出的 log 特徵）
    X_list, Y_list, ids_list = [], [], []
    for rec in records:
        kid = rec["kernel_id"]
        if kid not in id2vec:
            print(f"[warn] skip {kid} (no IR2Vec feature)")
            continue
        targets = np.array(rec["targets"], dtype=np.float32)
        # 對 NaN 以 fp32_err 或最大可用值填補，避免訓練 NaN
        if np.isnan(targets[:-1]).any():
            nan_mask = np.isnan(targets[:-1])
            fill_val = np.nanmax(targets[:-1]) if not np.isnan(targets[:-1]).all() else (
                targets[-1] if not np.isnan(targets[-1]) else 1.0
            )
            targets[:-1][nan_mask] = fill_val
        if np.isnan(targets[-1]):
            targets[-1] = np.nanmax(targets)

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

        # dot 統計特徵（推論時由 runtime 提供 mean/std/p1/p50/p99）
        if not rec.get("has_stats", False):
            stats_feats = [0.0] * (2 * STATS_DIM_PER_VEC)
        else:
            x_mean = float(rec.get("x_mean", 0.0))
            x_std = float(rec.get("x_std", 0.0))
            x_p1 = float(rec.get("x_p1", 0.0))
            x_p50 = float(rec.get("x_p50", 0.0))
            x_p99 = float(rec.get("x_p99", 0.0))

            y_mean = float(rec.get("y_mean", 0.0))
            y_std = float(rec.get("y_std", 0.0))
            y_p1 = float(rec.get("y_p1", 0.0))
            y_p50 = float(rec.get("y_p50", 0.0))
            y_p99 = float(rec.get("y_p99", 0.0))

            x_std_pos = float(x_std) if x_std > 0.0 else 0.0
            y_std_pos = float(y_std) if y_std > 0.0 else 0.0
            x_denom = x_std_pos + EPS
            y_denom = y_std_pos + EPS

            def sweet_feats(p1: float, p50: float, p99: float):
                p1a = abs(float(p1))
                p50a = abs(float(p50))
                p99a = abs(float(p99))
                log_p1 = np.log10(p1a + EPS)
                log_p50 = np.log10(p50a + EPS)
                log_p99 = np.log10(p99a + EPS)
                out = [
                    log_p50,           # center_decade
                    (log_p99 - log_p1) # span_decades
                ]
                for es in ES_LIST:
                    log_useed = float(USEED_LOG10[es])
                    upper_excess = max(0.0, log_p99 - log_useed)
                    lower_excess = max(0.0, (-log_useed) - log_p1)
                    out.extend([upper_excess, lower_excess])
                return out

            stats_feats = [
                # x (scale + shape)
                np.log10(x_denom),
                (x_p99 - x_p50) / x_denom,
                (x_p50 - x_p1) / x_denom,
                x_mean / x_denom,
            ] + sweet_feats(x_p1, x_p50, x_p99) + [
                # y (scale + shape)
                np.log10(y_denom),
                (y_p99 - y_p50) / y_denom,
                (y_p50 - y_p1) / y_denom,
                y_mean / y_denom,
            ] + sweet_feats(y_p1, y_p50, y_p99)

        cfg_feats = id2cfg.get(kid, np.zeros((cfg_dim,), dtype=np.float32))
        feat = np.concatenate([base_vec, np.array(extra + stats_feats, dtype=np.float32), cfg_feats])

        X_list.append(feat)
        Y_list.append(targets)
        ids_list.append(f"{kid}|len{rec['vec_len']}|dist{rec['dist_type']}|sid{rec['sample_id']}")

    if not X_list:
        raise RuntimeError("No data to build regression dataset.")

    X_out = np.stack(X_list, axis=0)
    Y_out = np.stack(Y_list, axis=0)

    np.savez(
        DATA_DIR / "ml_dataset_ir_errors.npz",
        X=X_out,
        Y=Y_out,
        ids=np.array(ids_list),
        target_names=np.array(TARGET_NAMES),
        meta=np.array({
            "ir_dim": X_all.shape[1],
            "vec_scale": vec_max,
            "rows_scale": rows_max,
            "cols_scale": cols_max,
            "cfg_dim": cfg_dim,
            "cfg_feat_names": list(CFG_FEAT_NAMES),
            "extra_dim": len(extra + stats_feats) + cfg_dim,
            "vec_lens": sorted(set(vec_vals)),
            "rows_vals": sorted(set(row_vals)),
            "cols_vals": sorted(set(col_vals)),
            "stats_feat_names": [
                "x_log10_std",
                "x_upper_tail",
                "x_lower_tail",
                "x_standardized_mean",
                "x_center_decade",
                "x_span_decades",
                "x_upper_excess_es0",
                "x_lower_excess_es0",
                "x_upper_excess_es1",
                "x_lower_excess_es1",
                "x_upper_excess_es2",
                "x_lower_excess_es2",
                "y_log10_std",
                "y_upper_tail",
                "y_lower_tail",
                "y_standardized_mean",
                "y_center_decade",
                "y_span_decades",
                "y_upper_excess_es0",
                "y_lower_excess_es0",
                "y_upper_excess_es1",
                "y_lower_excess_es1",
                "y_upper_excess_es2",
                "y_lower_excess_es2",
            ],
            "eps": EPS,
        }, dtype=object),
    )
    print("saved ml_dataset_ir_errors.npz")
    print("X shape:", X_out.shape)
    print("Y shape:", Y_out.shape)
    print("targets:", TARGET_NAMES)


if __name__ == "__main__":
    main()
