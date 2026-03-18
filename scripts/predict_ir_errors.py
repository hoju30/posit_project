# predict_ir_errors.py
# 對單一 IR 預測各數值格式（posit(8/16/32, es<=2)、fp32）相對 fp64 的相對誤差
import argparse
import numpy as np
import ir2vec
import joblib
from pathlib import Path

from split_loops_and_analyze_cfg import cfg_feature_vector_from_ll_path, CFG_FEAT_NAMES

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def load_ir2vec_vector(ir_path: str):
    init_obj = ir2vec.initEmbedding(ir_path, "fa", "p")
    v = init_obj.getProgramVector()
    return np.asarray(v, dtype=np.float32)

def log10p(x: float, eps: float) -> float:
    return float(np.log10(float(x) + eps))

def tail_features(mean: float, std: float, p1: float, p50: float, p99: float, eps: float):
    if (
        float(mean) == 0.0
        and float(std) == 0.0
        and float(p1) == 0.0
        and float(p50) == 0.0
        and float(p99) == 0.0
    ):
        return [0.0] * 12

    std_pos = float(std) if float(std) > 0.0 else 0.0
    denom = std_pos + eps

    # magnitude-based (posit sweet spot)
    p1a = abs(float(p1))
    p50a = abs(float(p50))
    p99a = abs(float(p99))
    log_p1 = log10p(p1a, eps)
    log_p50 = log10p(p50a, eps)
    log_p99 = log10p(p99a, eps)

    # useed = 2^(2^es), es in {0,1,2}
    useed_log10 = [np.log10(2.0), np.log10(4.0), np.log10(16.0)]
    sweet = [log_p50, (log_p99 - log_p1)]
    for log_useed in useed_log10:
        upper_excess = max(0.0, log_p99 - log_useed)
        lower_excess = max(0.0, (-log_useed) - log_p1)
        sweet.extend([upper_excess, lower_excess])

    return [
        log10p(std_pos, eps),               # log10(std + eps) (scale)
        (float(p99) - float(p50)) / denom,  # upper_tail
        (float(p50) - float(p1)) / denom,   # lower_tail
        float(mean) / denom,                # standardized_mean
        *sweet,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ir", help="LLVM IR file (.ll or .bc)")
    parser.add_argument(
        "--model",
        default=str(MODEL_DIR / "ir2vec_error_predictor.joblib"),
        help="regression model file",
    )
    parser.add_argument(
        "--save-csv",
        help="optional path to save predictions (target,pred)",
    )
    parser.add_argument(
        "--save-json",
        help="optional path to save predictions as JSON (pred_rel_err mapping)",
    )
    parser.add_argument("--vec-len", type=float, default=0.0, help="vector length (for dot/sum etc.)")
    parser.add_argument("--rows", type=float, default=0.0, help="rows (matvec)")
    parser.add_argument("--cols", type=float, default=0.0, help="cols (matvec)")
    parser.add_argument("--alpha", type=float, default=0.0, help="axpy/axpby: alpha (optional)")
    parser.add_argument("--beta", type=float, default=0.0, help="axpby: beta (optional)")
    # dot: runtime 統計（提供 mean/std/p1/p50/p99；模型內轉成 tail features）
    parser.add_argument("--x-mean", type=float, default=0.0, help="dot: mean(x)")
    parser.add_argument("--x-std", type=float, default=0.0, help="dot: std(x)")
    parser.add_argument("--x-p1", type=float, default=0.0, help="dot: p1(x)")
    parser.add_argument("--x-p50", type=float, default=0.0, help="dot: p50(x)")
    parser.add_argument("--x-p99", type=float, default=0.0, help="dot: p99(x)")
    parser.add_argument("--y-mean", type=float, default=0.0, help="dot: mean(y)")
    parser.add_argument("--y-std", type=float, default=0.0, help="dot: std(y)")
    parser.add_argument("--y-p1", type=float, default=0.0, help="dot: p1(y)")
    parser.add_argument("--y-p50", type=float, default=0.0, help="dot: p50(y)")
    parser.add_argument("--y-p99", type=float, default=0.0, help="dot: p99(y)")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    reg = bundle["model"]
    target_names = bundle["target_names"]
    meta = bundle.get("meta", None) or {}

    v = load_ir2vec_vector(args.ir).reshape(1, -1)
    # 附加特徵：vec_len/rows/cols + dot(x/y) 統計特徵
    ir_dim = meta.get("ir_dim", v.shape[1])
    vec_scale = meta.get("vec_scale", 1.0)
    rows_scale = meta.get("rows_scale", 1.0)
    cols_scale = meta.get("cols_scale", 1.0)
    vec_lens = meta.get("vec_lens", [])
    eps = float(meta.get("eps", 1e-30))
    cfg_dim = int(meta.get("cfg_dim", 0))
    extra_dim = int(meta.get("extra_dim", 5 + 24 + cfg_dim))

    extra = []
    # snap vec_len 到最近的訓練值（若提供列表）
    vec_val = args.vec_len
    if vec_lens:
        vec_val = min(vec_lens, key=lambda x: abs(x - args.vec_len))
    extra.append(vec_val / vec_scale if vec_scale else 0.0)

    extra.append(args.rows / rows_scale if rows_scale else 0.0)
    extra.append(args.cols / cols_scale if cols_scale else 0.0)
    extra.append(float(args.alpha))
    extra.append(float(args.beta))

    stats_feats = (
        tail_features(args.x_mean, args.x_std, args.x_p1, args.x_p50, args.x_p99, eps)
        + tail_features(args.y_mean, args.y_std, args.y_p1, args.y_p50, args.y_p99, eps)
    )

    if cfg_dim > 0:
        try:
            cfg_feats, _ = cfg_feature_vector_from_ll_path(args.ir, scc_mode="auto")
            cfg_feats = np.asarray(cfg_feats, dtype=np.float32).tolist()
        except Exception:
            cfg_feats = [0.0] * cfg_dim
    else:
        cfg_feats = []

    extra_tail_len = extra_dim - len(extra) - len(stats_feats) - len(cfg_feats)
    if extra_tail_len < 0:
        extra_tail_len = 0
    extra_tail = [0.0] * extra_tail_len
    feat = np.concatenate(
        [v.flatten(), np.array(extra + stats_feats + cfg_feats + extra_tail, dtype=np.float32)]
    ).reshape(1, -1)

    preds = reg.predict(feat)[0]  # shape = (num_targets,) without fp64

    # 還原目標尺度（若模型有記錄 transform）
    y_transform = bundle.get("y_transform")
    if y_transform and y_transform.get("type") == "log10":
        eps = float(y_transform.get("eps", 0.0))
        preds = (10.0 ** preds) - eps

    print("=== IR2Vec Error Predictor ===")
    print("IR file:", args.ir)
    print("Predicted relative errors (vs fp64):")
    for name, val in zip(target_names, preds):
        print(f"  {name:14s}: {val:.6e}")

    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            header = ["target", "pred"]
            w.writerow(header)
            for name, pred in zip(target_names, preds):
                w.writerow([name, pred])
        print("Saved predictions to", args.save_csv)

    if args.save_json:
        import json

        payload = {
            "ir": str(args.ir),
            "pred_rel_err": {name: float(val) for name, val in zip(target_names, preds)},
            "meta": {
                "vec_len": float(args.vec_len),
                "rows": float(args.rows),
                "cols": float(args.cols),
                "alpha": float(args.alpha),
                "beta": float(args.beta),
                "x_stats": {
                    "mean": float(args.x_mean),
                    "std": float(args.x_std),
                    "p1": float(args.x_p1),
                    "p50": float(args.x_p50),
                    "p99": float(args.x_p99),
                },
                "y_stats": {
                    "mean": float(args.y_mean),
                    "std": float(args.y_std),
                    "p1": float(args.y_p1),
                    "p50": float(args.y_p50),
                    "p99": float(args.y_p99),
                },
            },
        }
        Path(args.save_json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print("Saved predictions to", args.save_json)


if __name__ == "__main__":
    main()
