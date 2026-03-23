#!/usr/bin/env python3
"""
run_loop_pipeline.py

Single entrypoint for the per-loop (mixed-precision) regression workflow (MLIR input):

  MLIR
    -> (onnx-mlir-opt) posit.loop_id + outline scf.for -> __posit_seg_* + marker
    -> (mlir-opt + mlir-translate) LLVM IR (.ll)
    -> sanitize to LLVM16-friendly IR
    -> (llvm-extract-16 --func __posit_seg_*) loop segments
    -> (ir2vec + regression model) per-segment pred_rel_err
    -> map segments back to MLIR loop_id via marker call
    -> plan.json (keyed by loop_id)
    -> (onnx-mlir-opt) attach plan (and auto choose argmin)

Notes
-----
- This is an *analysis* pipeline. It does not rewrite types yet; it only produces/attaches a plan.
- It assumes you run with the project venv so `ir2vec` is importable.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAW_STAT_FIELDS = [
    "mean",
    "std",
    "min",
    "max",
    "abs_max",
    "skewness",
    "excess_kurtosis",
    "p01",
    "p50",
    "p99",
    "near_zero_ratio",
    "pos_ratio",
    "neg_ratio",
]


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}")


def require_bin(path_or_name: str) -> str:
    p = Path(path_or_name)
    if p.exists():
        return str(p)
    w = shutil.which(path_or_name)
    if w:
        return w
    raise SystemExit(f"missing tool: {path_or_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir", required=True, help="input MLIR file")
    ap.add_argument(
        "--out-dir",
        default="pipeline_out",
        help="output directory (relative to repo root by default)",
    )

    ap.add_argument(
        "--onnx-mlir-opt",
        default="/home/hoju/test/onnx-mlir/build/Release/bin/onnx-mlir-opt",
        help="onnx-mlir-opt binary (with posit passes registered)",
    )
    ap.add_argument(
        "--mlir-opt",
        default="/home/hoju/test/llvm-project/build/bin/mlir-opt",
        help="mlir-opt binary (used for lowering to LLVM dialect)",
    )
    ap.add_argument(
        "--mlir-translate",
        default="/home/hoju/test/llvm-project/build/bin/mlir-translate",
        help="mlir-translate binary (used for MLIR LLVM dialect -> LLVM IR)",
    )

    ap.add_argument("--opt", default="opt-16", help="LLVM opt (must match segment tooling; default: opt-16)")
    ap.add_argument(
        "--llvm-extract",
        default="llvm-extract-16",
        help="LLVM llvm-extract (must match segment tooling; default: llvm-extract-16)",
    )

    # Extra features (forwarded to predictor)
    ap.add_argument("--model", default=str(ROOT / "models" / "ir2vec_error_predictor.joblib"))
    ap.add_argument("--vec-len", type=float, default=0.0)
    ap.add_argument("--rows", type=float, default=0.0)
    ap.add_argument("--cols", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--format-features-json", default=None)
    for prefix in ("x", "y"):
        for name in RAW_STAT_FIELDS:
            flag_name = name.replace("_", "-")
            flags = [f"--{prefix}-{flag_name}", f"--{prefix}-{name}"]
            if name == "p01":
                flags.append(f"--{prefix}-p1")
            ap.add_argument(*flags, dest=f"{prefix}_{name}", type=float, default=0.0)

    ap.add_argument("--pick", default="min_error", choices=["min_error", "min_bits_under_tol"])
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--agg", default="max", choices=["max", "mean"], help="aggregation used when multiple segments map to one loop")
    args = ap.parse_args()

    mlir_in = Path(args.mlir)
    if not mlir_in.exists():
        raise SystemExit(f"MLIR not found: {mlir_in}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_mlir_opt = require_bin(args.onnx_mlir_opt)
    mlir_opt = require_bin(args.mlir_opt)
    mlir_translate = require_bin(args.mlir_translate)
    opt_bin = require_bin(args.opt)
    llvm_extract_bin = require_bin(args.llvm_extract)

    marked_mlir = out_dir / "marked.mlir"
    llvm_dialect_mlir = out_dir / "llvm_dialect.mlir"
    llvm_raw = out_dir / "raw.ll"
    llvm16_ll = out_dir / "llvm16.ll"
    segments_dir = out_dir / "segments"
    preds_json = out_dir / "predictions.json"
    plan_json = out_dir / "plan.json"
    annotated_mlir = out_dir / "annotated.mlir"

    # 1) Mark loops in MLIR: loop_id + outline segments + marker call
    marked_mlir.write_text(
        subprocess.check_output(
            [
                onnx_mlir_opt,
                str(mlir_in),
                "--posit-assign-loop-id",
                "--posit-outline-loop-segments",
                "--posit-insert-loop-marker",
            ],
            text=True,
        ),
        encoding="utf-8",
    )

    # 2) Lower marked MLIR -> LLVM dialect MLIR -> LLVM IR
    # mlir-opt prints to stdout; re-run capturing output to file (avoid depending on -o support differences)
    llvm_dialect_mlir.write_text(
        subprocess.check_output(
            [
                mlir_opt,
                str(marked_mlir),
                "--convert-scf-to-cf",
                "--convert-cf-to-llvm",
                "--convert-arith-to-llvm",
                "--convert-index-to-llvm",
                "--convert-func-to-llvm",
                "--finalize-memref-to-llvm",
                "--reconcile-unrealized-casts",
            ],
            text=True,
        ),
        encoding="utf-8",
    )
    llvm_raw.write_text(
        subprocess.check_output([mlir_translate, str(llvm_dialect_mlir), "--mlir-to-llvmir"], text=True),
        encoding="utf-8",
    )

    # 3) Sanitize to LLVM16-friendly IR for opt-16/ir2vec
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "tools" / "sanitize_llvm_ir_for_llvm16.py"),
            str(llvm_raw),
            "-o",
            str(llvm16_ll),
        ]
    )

    # 4) Per-loop-segment prediction
    pred_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "predict_ir_errors_by_loops.py"),
        str(llvm16_ll),
        "--split-mode",
        "outlined",
        "--include-regex",
        r"^__posit_seg_",
        "--segments-dir",
        str(segments_dir),
        "--force-split",
        "--opt",
        opt_bin,
        "--llvm-extract",
        llvm_extract_bin,
        "--model",
        str(args.model),
        "--vec-len",
        str(args.vec_len),
        "--rows",
        str(args.rows),
        "--cols",
        str(args.cols),
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
        "--pick",
        str(args.pick),
        "--tol",
        str(args.tol),
        "--save-json",
        str(preds_json),
    ]
    if args.format_features_json:
        pred_cmd.extend(["--format-features-json", str(args.format_features_json)])
    for prefix in ("x", "y"):
        for name in RAW_STAT_FIELDS:
            pred_cmd.extend([f"--{prefix}-{name.replace('_', '-')}", str(getattr(args, f"{prefix}_{name}"))])
    run(pred_cmd, cwd=ROOT)

    # 5) Map segments -> loop_id via marker, build plan.json
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "tools" / "build_loop_plan_from_marked_segments.py"),
            "--predictions-json",
            str(preds_json),
            "--mlir",
            str(mlir_in),
            "--onnx-mlir-opt",
            onnx_mlir_opt,
            "--agg",
            str(args.agg),
            "--out",
            str(plan_json),
        ],
        cwd=ROOT,
    )

    # 6) Attach plan back to MLIR (no marker insertion in final output)
    annotated_mlir.write_text(
        subprocess.check_output(
            [
                onnx_mlir_opt,
                str(mlir_in),
                "--posit-assign-loop-id",
                f"--posit-attach-plan=posit-plan-json={plan_json}",
            ],
            text=True,
        ),
        encoding="utf-8",
    )

    print("[ok] wrote:", str(marked_mlir))
    print("[ok] wrote:", str(llvm16_ll))
    print("[ok] wrote:", str(preds_json))
    print("[ok] wrote:", str(plan_json))
    print("[ok] wrote:", str(annotated_mlir))


if __name__ == "__main__":
    main()
