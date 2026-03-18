#!/usr/bin/env python3
"""
Predict per-format relative errors for each extracted loop segment (mixed-precision path).

Workflow:
  1) Split input IR into loop segments (default: extract outlined funcs by name)
  2) For each segment, run IR2Vec to get the 300-d program vector
  3) Concatenate extra features (vec_len/rows/cols/alpha/beta + runtime stats-derived features)
  4) Run the regression model to predict relative errors vs fp64 for each format
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

from split_loops_and_analyze_cfg import cfg_feature_vector_from_ll_path, split_ir_by_loops


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def load_ir2vec_vector(ir_path: str) -> np.ndarray:
    init_obj = ir2vec.initEmbedding(ir_path, "fa", "p")
    v = init_obj.getProgramVector()
    return np.asarray(v, dtype=np.float32)


def log10p(x: float, eps: float) -> float:
    return float(np.log10(float(x) + eps))


def tail_features(mean: float, std: float, p1: float, p50: float, p99: float, eps: float) -> list[float]:
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


@dataclass(frozen=True)
class RuntimeStats:
    x_mean: float
    x_std: float
    x_p1: float
    x_p50: float
    x_p99: float
    y_mean: float
    y_std: float
    y_p1: float
    y_p50: float
    y_p99: float


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
        x_mean=float(args.x_mean),
        x_std=float(args.x_std),
        x_p1=float(args.x_p1),
        x_p50=float(args.x_p50),
        x_p99=float(args.x_p99),
        y_mean=float(args.y_mean),
        y_std=float(args.y_std),
        y_p1=float(args.y_p1),
        y_p50=float(args.y_p50),
        y_p99=float(args.y_p99),
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
        if v is None:
            v = d2.get(key, None)
        if v is None:
            return fallback
        return float(v)

    stats = RuntimeStats(
        x_mean=pick_xy(x_d, x_def, "mean", base_stats.x_mean),
        x_std=pick_xy(x_d, x_def, "std", base_stats.x_std),
        x_p1=pick_xy(x_d, x_def, "p1", base_stats.x_p1),
        x_p50=pick_xy(x_d, x_def, "p50", base_stats.x_p50),
        x_p99=pick_xy(x_d, x_def, "p99", base_stats.x_p99),
        y_mean=pick_xy(y_d, y_def, "mean", base_stats.y_mean),
        y_std=pick_xy(y_d, y_def, "std", base_stats.y_std),
        y_p1=pick_xy(y_d, y_def, "p1", base_stats.y_p1),
        y_p50=pick_xy(y_d, y_def, "p50", base_stats.y_p50),
        y_p99=pick_xy(y_d, y_def, "p99", base_stats.y_p99),
    )

    return extra, stats


def build_feature(
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
    eps = float(meta.get("eps", 1e-30))
    cfg_dim = int(meta.get("cfg_dim", 0))
    extra_dim = int(meta.get("extra_dim", 5 + 24 + cfg_dim))

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
        tail_features(stats.x_mean, stats.x_std, stats.x_p1, stats.x_p50, stats.x_p99, eps)
        + tail_features(stats.y_mean, stats.y_std, stats.y_p1, stats.y_p50, stats.y_p99, eps)
    )

    if cfg_dim > 0 and ir_path_for_cfg:
        try:
            cfg_feats, _ = cfg_feature_vector_from_ll_path(ir_path_for_cfg, scc_mode="auto")
            cfg_feats = np.asarray(cfg_feats, dtype=np.float32).tolist()
        except Exception:
            cfg_feats = [0.0] * cfg_dim
    else:
        cfg_feats = []

    extra_tail_len = extra_dim - len(scalars) - len(stats_feats) - len(cfg_feats)
    if extra_tail_len < 0:
        extra_tail_len = 0
    extra_tail = [0.0] * extra_tail_len

    feat = np.concatenate(
        [v.flatten(), np.asarray(scalars + stats_feats + cfg_feats + extra_tail, dtype=np.float32)]
    ).reshape(1, -1)
    return feat


def pick_format(
    target_names: list[str],
    preds: np.ndarray,
    *,
    strategy: str,
    tol: float,
) -> str:
    # prefer smaller storage for "min_bits_under_tol"
    bits: dict[str, int] = {}
    for name in target_names:
        if name.startswith("posit_"):
            try:
                bits[name] = int(name.split("_")[1])
            except Exception:
                bits[name] = 9999
        elif name == "fp32":
            bits[name] = 32
        elif name == "fp16":
            bits[name] = 16
        else:
            bits[name] = 9999

    if strategy == "min_error":
        idx = int(np.argmin(preds))
        return target_names[idx]

    if strategy == "min_bits_under_tol":
        candidates = [(bits[n], float(p), n) for n, p in zip(target_names, preds) if float(p) <= tol]
        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1]))
            return candidates[0][2]
        idx = int(np.argmin(preds))
        return target_names[idx]

    raise SystemExit(f"Unknown --pick strategy: {strategy}")


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
    p.add_argument("--x-mean", type=float, default=0.0)
    p.add_argument("--x-std", type=float, default=0.0)
    p.add_argument("--x-p1", type=float, default=0.0)
    p.add_argument("--x-p50", type=float, default=0.0)
    p.add_argument("--x-p99", type=float, default=0.0)
    p.add_argument("--y-mean", type=float, default=0.0)
    p.add_argument("--y-std", type=float, default=0.0)
    p.add_argument("--y-p1", type=float, default=0.0)
    p.add_argument("--y-p50", type=float, default=0.0)
    p.add_argument("--y-p99", type=float, default=0.0)

    p.add_argument(
        "--stats-json",
        default=None,
        help="optional per-segment overrides JSON (keys: default + segments{segment_id|func_name: {...}})",
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
    base_extra, base_stats = stats_from_args(args)

    bundle = joblib.load(args.model)
    reg = bundle["model"]
    target_names = bundle["target_names"]
    meta = bundle.get("meta", None) or {}

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
        feat = build_feature(ir_vec, str(seg_path), extra, stats, meta)

        pred = reg.predict(feat)[0]  # log10-space if trained that way
        pred = np.asarray(pred, dtype=np.float64)
        if y_is_log10:
            pred = (10.0 ** pred) - y_eps

        chosen = pick_format(target_names, pred, strategy=str(args.pick), tol=float(args.tol))
        chosen_pred = float(pred[list(target_names).index(chosen)])

        pred_dict = {str(n): float(v) for n, v in zip(target_names, pred)}
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
                    "x": {
                        "mean": float(stats.x_mean),
                        "std": float(stats.x_std),
                        "p1": float(stats.x_p1),
                        "p50": float(stats.x_p50),
                        "p99": float(stats.x_p99),
                    },
                    "y": {
                        "mean": float(stats.y_mean),
                        "std": float(stats.y_std),
                        "p1": float(stats.y_p1),
                        "p50": float(stats.y_p50),
                        "p99": float(stats.y_p99),
                    },
                },
                "pred_rel_err": pred_dict,
                "chosen": chosen,
                "chosen_pred": chosen_pred,
            }
        )
        pred_mat.append(pred)

    pred_mat_np = np.stack(pred_mat, axis=0)  # (num_segments, num_targets)
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
            "pred_rel_err": {str(n): float(v) for n, v in zip(target_names, agg)},
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
    for name, val in zip(target_names, agg):
        print(f"  {name:14s}: {float(val):.6e}")

    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["target", "pred"])
            for name, val in zip(target_names, agg):
                w.writerow([name, float(val)])
        print("Saved aggregate predictions to", args.save_csv)

    if args.save_segments_csv:
        with open(args.save_segments_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["segment_id", "func_name", "target", "pred", "chosen"])
            for seg in per_seg:
                chosen = seg["chosen"]
                for t in target_names:
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
