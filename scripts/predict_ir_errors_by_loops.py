#!/usr/bin/env python3
"""
Predict per-format relative errors for each extracted loop segment (mixed-precision path).

Workflow:
  1) Split input IR into loop segments (default: extract outlined funcs by name)
  2) For each segment, run IR2Vec to get the 300-d program vector
  3) Concatenate shared features (vec_len/rows/cols/alpha/beta + runtime stats + CFG)
  4) For each posit format, append format features (n/es) and run the per-format regressor
  5) Optionally pick a per-segment format (mixed precision) and write a JSON plan

This script does NOT rewrite IR. It produces a decision plan you can feed into an LLVM/MLIR pass later.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ir2vec
import joblib
import numpy as np

from error_feature_utils import (
    RAW_STAT_FIELDS,
    derived_stats_features,
    parse_format_name,
    raw_stats_features,
)
from posit_test.scripts.split_loops import split_ir_by_loops


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
def load_ir2vec_vector(ir_path: str) -> np.ndarray:
    init_obj = ir2vec.initEmbedding(ir_path, "fa", "p")
    v = init_obj.getProgramVector()
    return np.asarray(v, dtype=np.float32)


def parse_format_features_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--format-features-json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RuntimeStats:
    x: dict[str, float]
    y: dict[str, float]


@dataclass(frozen=True)
class ExtraScalars:
    vec_len: float
    rows: float
    cols: float
    alpha: float
    beta: float


def split_if_needed(
    input_ir: Path,
    out_dir: Path,
    *,
    force: bool,
    split_mode: str,
    opt_bin: str,
    llvm_extract_bin: str,
    include_regex: str,
    recursive: bool,
    allow_no_extract: bool,
) -> Path:
    return split_ir_by_loops(
        input_ir,
        out_dir=out_dir,
        force=force,
        split_mode=split_mode,
        opt=opt_bin,
        llvm_extract=llvm_extract_bin,
        include_regex=include_regex,
        recursive=recursive,
        allow_no_extract=allow_no_extract,
    )


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    if manifest_path.suffix.lower() == ".json":
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    # Backward-compat for older runs
    if manifest_path.suffix.lower() == ".csv":
        segments = []
        with open(manifest_path, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                segments.append(
                    {
                        "segment_id": int(row["segment_id"]),
                        "func_name": row["func_name"],
                        "path": str(Path(row["path"]).resolve()),
                    }
                )
        return {
            "input_ir": None,
            "out_dir": str(manifest_path.parent.resolve()),
            "include_regex": None,
            "segments": segments,
        }

    raise SystemExit(f"Unsupported manifest format: {manifest_path}")


def parse_stats_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--stats-json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def stats_from_args(args: argparse.Namespace) -> tuple[ExtraScalars, RuntimeStats]:
    extra = ExtraScalars(
        vec_len=float(args.vec_len),
        rows=float(args.rows),
        cols=float(args.cols),
        alpha=float(args.alpha),
        beta=float(args.beta),
    )
    stats = RuntimeStats(
        x={name: float(getattr(args, f"x_{name}")) for name in RAW_STAT_FIELDS},
        y={name: float(getattr(args, f"y_{name}")) for name in RAW_STAT_FIELDS},
    )
    return extra, stats


def apply_overrides(
    seg: dict[str, Any],
    base_extra: ExtraScalars,
    base_stats: RuntimeStats,
    overrides: dict[str, Any],
) -> tuple[ExtraScalars, RuntimeStats]:
    if not overrides:
        return base_extra, base_stats

    def get_seg_override() -> dict[str, Any]:
        seg_id = str(seg.get("segment_id", ""))
        fn = str(seg.get("func_name", ""))
        segs = overrides.get("segments", {})
        if isinstance(segs, dict):
            if seg_id in segs:
                return segs[seg_id]
            if fn in segs:
                return segs[fn]
        return {}

    default = overrides.get("default", {}) if isinstance(overrides.get("default", {}), dict) else {}
    spec = get_seg_override()

    def pick(d: dict[str, Any], key: str, fallback: float) -> float:
        v = d.get(key, None)
        if v is None:
            v = default.get(key, None)
        if v is None:
            return fallback
        return float(v)

    extra = ExtraScalars(
        vec_len=pick(spec, "vec_len", base_extra.vec_len),
        rows=pick(spec, "rows", base_extra.rows),
        cols=pick(spec, "cols", base_extra.cols),
        alpha=pick(spec, "alpha", base_extra.alpha),
        beta=pick(spec, "beta", base_extra.beta),
    )

    # Allow nested form: { "x": {...}, "y": {...} }
    x_d = spec.get("x", {}) if isinstance(spec.get("x", {}), dict) else {}
    y_d = spec.get("y", {}) if isinstance(spec.get("y", {}), dict) else {}
    x_def = default.get("x", {}) if isinstance(default.get("x", {}), dict) else {}
    y_def = default.get("y", {}) if isinstance(default.get("y", {}), dict) else {}

    def pick_xy(d: dict[str, Any], d2: dict[str, Any], key: str, fallback: float) -> float:
        v = d.get(key, None)
        if v is None and key == "p01":
            v = d.get("p1", None)
        if v is None:
            v = d2.get(key, None)
        if v is None and key == "p01":
            v = d2.get("p1", None)
        if v is None:
            return fallback
        return float(v)

    stats = RuntimeStats(
        x={name: pick_xy(x_d, x_def, name, base_stats.x.get(name, 0.0)) for name in RAW_STAT_FIELDS},
        y={name: pick_xy(y_d, y_def, name, base_stats.y.get(name, 0.0)) for name in RAW_STAT_FIELDS},
    )

    return extra, stats


def apply_format_feature_overrides(
    seg: dict[str, Any],
    overrides: dict[str, Any],
    quant_feat_names: list[str],
    format_names: list[str],
) -> dict[str, dict[str, float]]:
    if not overrides:
        return {fmt: {name: 0.0 for name in quant_feat_names} for fmt in format_names}

    if "default" not in overrides and "segments" not in overrides:
        return {
            str(fmt): {name: float(overrides.get(str(fmt), {}).get(name, 0.0)) for name in quant_feat_names}
            for fmt in format_names
        }

    seg_id = str(seg.get("segment_id", ""))
    fn = str(seg.get("func_name", ""))
    segs = overrides.get("segments", {})
    seg_spec = {}
    if isinstance(segs, dict):
        if seg_id in segs:
            seg_spec = segs[seg_id]
        elif fn in segs:
            seg_spec = segs[fn]
    default = overrides.get("default", {}) if isinstance(overrides.get("default", {}), dict) else {}

    out: dict[str, dict[str, float]] = {}
    for fmt in format_names:
        fmt_default = default.get(str(fmt), {}) if isinstance(default.get(str(fmt), {}), dict) else {}
        fmt_seg = seg_spec.get(str(fmt), {}) if isinstance(seg_spec.get(str(fmt), {}), dict) else {}
        out[str(fmt)] = {
            name: float(fmt_seg.get(name, fmt_default.get(name, 0.0)))
            for name in quant_feat_names
        }
    return out


def build_shared_feature(
    ir_vec: np.ndarray,
    ir_path_for_cfg: str | None,
    extra: ExtraScalars,
    stats: RuntimeStats,
    meta: dict[str, Any],
) -> np.ndarray:
    v = ir_vec.reshape(1, -1)

    vec_scale = float(meta.get("vec_scale", 1.0))
    rows_scale = float(meta.get("rows_scale", 1.0))
    cols_scale = float(meta.get("cols_scale", 1.0))
    vec_lens = meta.get("vec_lens", [])
    cfg_dim = int(meta.get("cfg_dim", 0))
    shared_feat_dim = int(meta.get("shared_feat_dim", 5 + 2 * len(RAW_STAT_FIELDS) + cfg_dim))

    vec_val = float(extra.vec_len)
    if vec_lens:
        vec_val = min(vec_lens, key=lambda x: abs(float(x) - float(extra.vec_len)))

    scalars = [
        vec_val / vec_scale if vec_scale else 0.0,
        float(extra.rows) / rows_scale if rows_scale else 0.0,
        float(extra.cols) / cols_scale if cols_scale else 0.0,
        float(extra.alpha),
        float(extra.beta),
    ]
    stats_feats = (
        raw_stats_features(stats.x)
        + raw_stats_features(stats.y)
        + derived_stats_features(stats.x)
        + derived_stats_features(stats.y)
    )

    cfg_feats = []

    extra_tail_len = shared_feat_dim - len(scalars) - len(stats_feats) - len(cfg_feats)
    if extra_tail_len < 0:
        extra_tail_len = 0
    extra_tail = [0.0] * extra_tail_len

    feat = np.concatenate(
        [v.flatten(), np.asarray(scalars + stats_feats + cfg_feats + extra_tail, dtype=np.float32)]
    )
    return feat


def build_format_feature(shared_feat: np.ndarray, n: int, es: int, meta: dict[str, Any]) -> np.ndarray:
    format_feat_dim = int(meta.get("format_feat_dim", 2))
    format_feats = [float(n) / 32.0, float(es) / 2.0]
    if format_feat_dim > len(format_feats):
        format_feats.extend([0.0] * (format_feat_dim - len(format_feats)))
    return np.concatenate([shared_feat, np.asarray(format_feats[:format_feat_dim], dtype=np.float32)]).reshape(1, -1)


def pick_format(
    format_names: list[str],
    preds: np.ndarray,
    *,
    strategy: str,
    tol: float,
) -> str:
    # prefer smaller storage for "min_bits_under_tol"
    bits: dict[str, int] = {}
    for name in format_names:
        if name.startswith("posit_"):
            try:
                bits[name] = int(name.split("_")[1])
            except Exception:
                bits[name] = 9999
        else:
            bits[name] = 9999

    if strategy == "min_error":
        idx = int(np.argmin(preds))
        return format_names[idx]

    if strategy == "min_bits_under_tol":
        candidates = [(bits[n], float(p), n) for n, p in zip(format_names, preds) if float(p) <= tol]
        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1]))
            return candidates[0][2]
        idx = int(np.argmin(preds))
        return format_names[idx]

    raise SystemExit(f"Unknown --pick strategy: {strategy}")


def apply_selective_calibration(preds: np.ndarray, format_names: list[str], bundle: dict[str, Any]) -> np.ndarray:
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ir", help="input LLVM IR (.ll)")
    p.add_argument(
        "--model",
        default=str(MODEL_DIR / "ir2vec_error_predictor.joblib"),
        help="regression model file",
    )
    p.add_argument(
        "--split-mode",
        default="outlined",
        choices=["outlined", "loop-extract"],
        help="segmentation mode: outlined funcs in IR (default) or LLVM loop-extract",
    )
    p.add_argument(
        "--segments-dir",
        default=None,
        help="directory to store/read loop segments (default: ir/segments/<stem>)",
    )
    p.add_argument("--verbose", action="store_true", help="print progress while running")
    p.add_argument("--force-split", action="store_true", help="re-run loop splitting even if manifest exists")
    p.add_argument(
        "--allow-no-extract",
        action="store_true",
        help="if no segment functions are matched, fall back to a single segment (debug only)",
    )
    p.add_argument("--opt", default="opt-16", help="opt binary (default: opt-16)")
    p.add_argument("--llvm-extract", default="llvm-extract-16", help="llvm-extract binary (default: llvm-extract-16)")
    p.add_argument(
        "--include-regex",
        default=r"^__posit_seg_",
        help=r"regex for segment function names (default: ^__posit_seg_)",
    )
    p.add_argument("--recursive", action="store_true", help="pass --recursive to llvm-extract")

    # Global extra scalars (applied to all segments unless overridden)
    p.add_argument("--vec-len", type=float, default=0.0)
    p.add_argument("--rows", type=float, default=0.0)
    p.add_argument("--cols", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=0.0)

    # Global runtime stats (applied to all segments unless overridden)
    for prefix in ("x", "y"):
        for name in RAW_STAT_FIELDS:
            flag_name = name.replace("_", "-")
            flags = [f"--{prefix}-{flag_name}", f"--{prefix}-{name}"]
            if name == "p01":
                flags.append(f"--{prefix}-p1")
            p.add_argument(*flags, dest=f"{prefix}_{name}", type=float, default=0.0)

    p.add_argument(
        "--stats-json",
        default=None,
        help="optional per-segment overrides JSON (keys: default + segments{segment_id|func_name: {...}})",
    )
    p.add_argument(
        "--format-features-json",
        default=None,
        help="optional per-segment/per-format quant feature JSON",
    )

    p.add_argument(
        "--pick",
        default="min_bits_under_tol",
        choices=["min_error", "min_bits_under_tol"],
        help="per-segment format selection strategy",
    )
    p.add_argument("--tol", type=float, default=1e-3, help="tolerance for --pick=min_bits_under_tol")

    p.add_argument(
        "--save-json",
        default=None,
        help="save per-segment predictions/choices to JSON (default: <segments-dir>/predictions.json)",
    )
    p.add_argument(
        "--save-csv",
        default=None,
        help="save aggregated predictions to CSV (target,pred) for plot_bar.py",
    )
    p.add_argument(
        "--aggregate",
        default="max",
        choices=["max", "mean"],
        help="aggregation method used for --save-csv and summary",
    )
    p.add_argument(
        "--save-segments-csv",
        default=None,
        help="optional long-form CSV: segment_id,func_name,target,pred,chosen",
    )

    args = p.parse_args()

    in_path = Path(args.ir)
    if not in_path.exists():
        raise SystemExit(f"IR not found: {in_path}")

    segments_dir = Path(args.segments_dir) if args.segments_dir else (ROOT / "ir" / "segments" / in_path.stem)
    if args.verbose:
        print("[info] input IR      :", str(in_path), flush=True)
        print("[info] segments dir  :", str(segments_dir), flush=True)
    manifest_path = split_if_needed(
        in_path,
        segments_dir,
        force=bool(args.force_split),
        split_mode=str(args.split_mode),
        opt_bin=str(args.opt),
        llvm_extract_bin=str(args.llvm_extract),
        include_regex=str(args.include_regex),
        recursive=bool(args.recursive),
        allow_no_extract=bool(args.allow_no_extract),
    )
    if args.verbose:
        print("[info] manifest      :", str(manifest_path), flush=True)

    manifest = load_manifest(manifest_path)
    segments = manifest.get("segments", [])
    if not segments:
        raise SystemExit("No segments in manifest; nothing to predict.")
    if args.verbose:
        print("[info] segments      :", len(segments), flush=True)

    overrides = parse_stats_json(args.stats_json)
    format_feature_overrides = parse_format_features_json(args.format_features_json)
    base_extra, base_stats = stats_from_args(args)

    bundle = joblib.load(args.model)
    reg = bundle.get("model")
    reg_map = bundle.get("models")
    format_names = bundle.get("format_names")
    if format_names is None:
        raise SystemExit("model missing format_names; retrain with per-format regression dataset")
    meta = bundle.get("meta", None) or {}
    quant_feat_names = list(meta.get("quant_feat_names", []))

    y_transform = bundle.get("y_transform")
    y_is_log10 = bool(y_transform and y_transform.get("type") == "log10")
    y_eps = float(y_transform.get("eps", 0.0)) if y_is_log10 else 0.0

    per_seg: list[dict[str, Any]] = []
    pred_mat = []

    for idx, seg in enumerate(segments):
        seg_path = Path(seg["path"])
        if args.verbose:
            fn = str(seg.get("func_name", seg_path.name))
            print(f"[info] segment {idx+1}/{len(segments)}: {fn}", flush=True)
        ir_vec = load_ir2vec_vector(str(seg_path))
        extra, stats = apply_overrides(seg, base_extra, base_stats, overrides)
        shared_feat = build_shared_feature(ir_vec, str(seg_path), extra, stats, meta)
        format_feature_map = apply_format_feature_overrides(
            seg, format_feature_overrides, quant_feat_names, list(format_names)
        )
        feat_mat = []
        for name in format_names:
            n, es = parse_format_name(str(name))
            feat = build_format_feature(shared_feat, n, es, meta).reshape(-1)
            if quant_feat_names:
                quant_vals = np.asarray(
                    [float(format_feature_map.get(str(name), {}).get(qname, 0.0)) for qname in quant_feat_names],
                    dtype=np.float32,
                )
                feat[-len(quant_feat_names):] = quant_vals
            feat_mat.append(feat)
        feat_mat_np = np.stack(feat_mat, axis=0)

        if reg_map:
            pred = np.asarray(
                [reg_map[str(name)].predict(feat_mat_np[i : i + 1])[0] for i, name in enumerate(format_names)]
            )
        elif reg is not None:
            pred = reg.predict(feat_mat_np)
        else:
            raise SystemExit("model file missing both 'model' and 'models'")
        pred = np.asarray(pred, dtype=np.float64)
        if y_is_log10:
            pred = (10.0 ** pred) - y_eps
        pred = apply_selective_calibration(pred, list(format_names), bundle)

        chosen = pick_format(format_names, pred, strategy=str(args.pick), tol=float(args.tol))
        chosen_pred = float(pred[list(format_names).index(chosen)])

        pred_dict = {str(n): float(v) for n, v in zip(format_names, pred)}
        per_seg.append(
            {
                "segment_id": int(seg.get("segment_id", -1)),
                "func_name": str(seg.get("func_name", "")),
                "path": str(seg_path.resolve()),
                "extra": {
                    "vec_len": float(extra.vec_len),
                    "rows": float(extra.rows),
                    "cols": float(extra.cols),
                    "alpha": float(extra.alpha),
                    "beta": float(extra.beta),
                },
                "runtime_stats": {
                    "x": {name: float(stats.x.get(name, 0.0)) for name in RAW_STAT_FIELDS},
                    "y": {name: float(stats.y.get(name, 0.0)) for name in RAW_STAT_FIELDS},
                },
                "format_features": format_feature_map,
                "pred_rel_err": pred_dict,
                "chosen": chosen,
                "chosen_pred": chosen_pred,
            }
        )
        pred_mat.append(pred)

    pred_mat_np = np.stack(pred_mat, axis=0)  # (num_segments, num_formats)
    agg_fn = np.max if args.aggregate == "max" else np.mean
    agg = agg_fn(pred_mat_np, axis=0)

    out_json = Path(args.save_json) if args.save_json else (segments_dir / "predictions.json")
    payload = {
        "input_ir": str(in_path.resolve()),
        "model": str(Path(args.model).resolve()),
        "segments_dir": str(segments_dir.resolve()),
        "manifest": str(Path(manifest_path).resolve()),
        "pick": {"strategy": str(args.pick), "tol": float(args.tol)},
        "summary": {
            "aggregate": str(args.aggregate),
            "pred_rel_err": {str(n): float(v) for n, v in zip(format_names, agg)},
        },
        "segments": per_seg,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=== IR2Vec Error Predictor (By Loops) ===")
    print("IR file     :", str(in_path))
    print("segments    :", len(per_seg))
    print("manifest    :", str(manifest_path))
    print("saved json  :", str(out_json))
    print(f"aggregate   : {args.aggregate}")
    print("Agg predicted relative errors (vs fp64):")
    for name, val in zip(format_names, agg):
        print(f"  {name:14s}: {float(val):.6e}")

    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["target", "pred"])
            for name, val in zip(format_names, agg):
                w.writerow([name, float(val)])
        print("Saved aggregate predictions to", args.save_csv)

    if args.save_segments_csv:
        with open(args.save_segments_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["segment_id", "func_name", "target", "pred", "chosen"])
            for seg in per_seg:
                chosen = seg["chosen"]
                for t in format_names:
                    w.writerow(
                        [
                            seg["segment_id"],
                            seg["func_name"],
                            t,
                            seg["pred_rel_err"][t],
                            1 if t == chosen else 0,
                        ]
                    )
        print("Saved per-segment predictions to", args.save_segments_csv)


if __name__ == "__main__":
    main()
