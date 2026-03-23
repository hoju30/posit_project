# predict_ir_errors.py
# 對單一 IR 預測各數值格式（posit(8/16/32, es<=2)、fp32）相對 fp64 的相對誤差
import argparse
import numpy as np
import ir2vec
import joblib
from pathlib import Path

from error_feature_utils import (
    RAW_STAT_FIELDS,
    derived_stats_features,
    parse_format_name,
    raw_stats_features,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def load_ir2vec_vector(ir_path: str):
    init_obj = ir2vec.initEmbedding(ir_path, "fa", "p")
    v = init_obj.getProgramVector()
    return np.asarray(v, dtype=np.float32)


def parse_format_features_json(path: str | None) -> dict[str, dict[str, float]]:
    if not path:
        return {}
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload.get("formats"), dict):
        payload = payload["formats"]
    out: dict[str, dict[str, float]] = {}
    for fmt, vals in payload.items():
        if isinstance(vals, dict):
            out[str(fmt)] = {str(k): float(v) for k, v in vals.items()}
    return out


def apply_selective_calibration(preds: np.ndarray, format_names: list[str], bundle: dict) -> np.ndarray:
    calib = bundle.get("calibration")
    if not calib or not bool(calib.get("enabled", False)):
        return preds
    if str(calib.get("type")) != "log_linear_selective":
        return preds
    out = np.asarray(preds, dtype=np.float64).copy()
    for i, name in enumerate(format_names):
        spec = calib.get("formats", {}).get(str(name), {})
        if not bool(spec.get("use_calibrated", False)):
            continue
        slope = float(spec.get("slope", 1.0))
        intercept = float(spec.get("intercept", 0.0))
        pred_log = np.log10(max(float(out[i]), 0.0) + 1e-30)
        out[i] = max((10.0 ** (pred_log * slope + intercept)) - 1e-30, 0.0)
    return out


def build_shared_feature(
    ir_path: str,
    ir_vec: np.ndarray,
    args: argparse.Namespace,
    meta: dict[str, object],
) -> np.ndarray:
    v = ir_vec.reshape(1, -1)
    vec_scale = float(meta.get("vec_scale", 1.0))
    rows_scale = float(meta.get("rows_scale", 1.0))
    cols_scale = float(meta.get("cols_scale", 1.0))
    vec_lens = meta.get("vec_lens", [])
    cfg_dim = int(meta.get("cfg_dim", 0))
    shared_feat_dim = int(meta.get("shared_feat_dim", 5 + 2 * len(RAW_STAT_FIELDS) + cfg_dim))

    extra = []
    vec_val = args.vec_len
    if vec_lens:
        vec_val = min(vec_lens, key=lambda x: abs(x - args.vec_len))
    extra.append(vec_val / vec_scale if vec_scale else 0.0)
    extra.append(args.rows / rows_scale if rows_scale else 0.0)
    extra.append(args.cols / cols_scale if cols_scale else 0.0)
    extra.append(float(args.alpha))
    extra.append(float(args.beta))

    x_stats_map = {name: float(getattr(args, f"x_{name}")) for name in RAW_STAT_FIELDS}
    y_stats_map = {name: float(getattr(args, f"y_{name}")) for name in RAW_STAT_FIELDS}
    stats_feats = (
        raw_stats_features(x_stats_map)
        + raw_stats_features(y_stats_map)
        + derived_stats_features(x_stats_map)
        + derived_stats_features(y_stats_map)
    )

    cfg_feats = []

    extra_tail_len = shared_feat_dim - len(extra) - len(stats_feats) - len(cfg_feats)
    if extra_tail_len < 0:
        extra_tail_len = 0
    extra_tail = [0.0] * extra_tail_len
    return np.concatenate(
        [v.flatten(), np.array(extra + stats_feats + cfg_feats + extra_tail, dtype=np.float32)]
    )


def build_format_feature(shared_feat: np.ndarray, n: int, es: int, meta: dict[str, object]) -> np.ndarray:
    format_feat_dim = int(meta.get("format_feat_dim", 2))
    format_feats = [float(n) / 32.0, float(es) / 2.0]
    if format_feat_dim > len(format_feats):
        format_feats.extend([0.0] * (format_feat_dim - len(format_feats)))
    return np.concatenate([shared_feat, np.asarray(format_feats[:format_feat_dim], dtype=np.float32)]).reshape(1, -1)


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
    parser.add_argument(
        "--format-features-json",
        help="optional JSON mapping per format quant features, e.g. {\"posit_8_0\": {\"x_oor_ratio\": ...}}",
    )
    parser.add_argument("--vec-len", type=float, default=0.0, help="vector length (for dot/sum etc.)")
    parser.add_argument("--rows", type=float, default=0.0, help="rows (matvec)")
    parser.add_argument("--cols", type=float, default=0.0, help="cols (matvec)")
    parser.add_argument("--alpha", type=float, default=0.0, help="axpy/axpby: alpha (optional)")
    parser.add_argument("--beta", type=float, default=0.0, help="axpby: beta (optional)")
    for prefix in ("x", "y"):
        for name in RAW_STAT_FIELDS:
            flag_name = name.replace("_", "-")
            flags = [f"--{prefix}-{flag_name}", f"--{prefix}-{name}"]
            if name == "p01":
                flags.append(f"--{prefix}-p1")
            parser.add_argument(*flags, dest=f"{prefix}_{name}", type=float, default=0.0)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    reg = bundle.get("model")
    reg_map = bundle.get("models")
    format_names = bundle.get("format_names")
    if format_names is None:
        raise SystemExit("model missing format_names; retrain with per-format regression dataset")
    meta = bundle.get("meta", None) or {}
    quant_feat_names = list(meta.get("quant_feat_names", []))
    format_feature_map = parse_format_features_json(args.format_features_json)

    ir_vec = load_ir2vec_vector(args.ir)
    shared_feat = build_shared_feature(args.ir, ir_vec, args, meta)
    feat_mat = []
    for name in format_names:
        n, es = parse_format_name(str(name))
        feat = build_format_feature(shared_feat, n, es, meta).reshape(-1)
        if quant_feat_names:
            fmt_map = format_feature_map.get(str(name), {})
            quant_vals = np.asarray([float(fmt_map.get(qname, 0.0)) for qname in quant_feat_names], dtype=np.float32)
            feat[-len(quant_feat_names):] = quant_vals
        feat_mat.append(feat)
    feat_mat_np = np.stack(feat_mat, axis=0)
    if reg_map:
        preds = np.asarray([reg_map[str(name)].predict(feat_mat_np[i : i + 1])[0] for i, name in enumerate(format_names)])
    elif reg is not None:
        preds = reg.predict(feat_mat_np)
    else:
        raise SystemExit("model file missing both 'model' and 'models'")

    # 還原目標尺度（若模型有記錄 transform）
    y_transform = bundle.get("y_transform")
    if y_transform and y_transform.get("type") == "log10":
        eps = float(y_transform.get("eps", 0.0))
        preds = (10.0 ** preds) - eps
    preds = apply_selective_calibration(np.asarray(preds, dtype=np.float64), list(format_names), bundle)

    print("=== IR2Vec Error Predictor ===")
    print("IR file:", args.ir)
    print("Predicted relative errors (vs fp64):")
    for name, val in zip(format_names, preds):
        print(f"  {name:14s}: {val:.6e}")

    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            header = ["target", "pred"]
            w.writerow(header)
            for name, pred in zip(format_names, preds):
                w.writerow([name, pred])
        print("Saved predictions to", args.save_csv)

    if args.save_json:
        import json

        payload = {
            "ir": str(args.ir),
            "pred_rel_err": {str(name): float(val) for name, val in zip(format_names, preds)},
            "meta": {
                "vec_len": float(args.vec_len),
                "rows": float(args.rows),
                "cols": float(args.cols),
                "alpha": float(args.alpha),
                "beta": float(args.beta),
                "x_stats": {name: float(getattr(args, f"x_{name}")) for name in RAW_STAT_FIELDS},
                "y_stats": {name: float(getattr(args, f"y_{name}")) for name in RAW_STAT_FIELDS},
                "format_features_json": str(args.format_features_json) if args.format_features_json else None,
            },
        }
        Path(args.save_json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print("Saved predictions to", args.save_json)


if __name__ == "__main__":
    main()
