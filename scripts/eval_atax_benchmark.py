#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "benchmark_atax_eval.cpp"
BIN = ROOT / "bin" / "atax_eval"
IR = Path("/home/hoju/test/llvm-project/polly/test/ForwardOpTree/atax.ll")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def compile_binary() -> None:
    BIN.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "g++",
            "-std=c++20",
            "-O2",
            str(SRC),
            "-I" + str(ROOT / "external" / "universal" / "include" / "sw"),
            "-o",
            str(BIN),
        ]
    )


def fmt_stat_args(prefix: str, stats: dict[str, float]) -> list[str]:
    out: list[str] = []
    for k, v in stats.items():
        out.append(f"--{prefix}-{k.replace('_', '-')}={v}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--cols", type=int, default=4000)
    ap.add_argument("--dist-type", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--polybench-init", action="store_true")
    ap.add_argument("--formats", default="", help="optional subset, e.g. 16:1,16:2,32:0,32:1,32:2")
    ap.add_argument("--model", default=str(ROOT / "models" / "ir2vec_error_predictor.joblib"))
    ap.add_argument("--ir", default="/home/hoju/test/polybench-c-3.2/atax.ll")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "atax_benchmark_eval"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actual_json = out_dir / "actual.json"
    pred_json = out_dir / "pred.json"
    format_json = out_dir / "format_features.json"
    compare_json = out_dir / "compare.json"
    compare_csv = out_dir / "compare.csv"

    compile_binary()

    run(
        [
            str(BIN),
            "--rows",
            str(args.rows),
            "--cols",
            str(args.cols),
            "--dist-type",
            str(args.dist_type),
            "--seed",
            str(args.seed),
            "--out-json",
            str(actual_json),
        ]
        + (["--polybench-init"] if args.polybench_init else [])
        + (["--formats", args.formats] if args.formats else [])
    )

    actual = json.loads(actual_json.read_text(encoding="utf-8"))
    x_stats = actual["x_stats"]
    y_stats = actual["y_stats"]
    format_json.write_text(
        json.dumps(actual.get("format_features", {"formats": {}}), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    pred_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "predict_ir_errors.py"),
        str(args.ir),
        "--model",
        str(args.model),
        "--rows",
        str(args.rows),
        "--cols",
        str(args.cols),
        "--vec-len",
        str(args.cols),
        "--format-features-json",
        str(format_json),
        "--save-json",
        str(pred_json),
    ]
    pred_cmd.extend(fmt_stat_args("x", x_stats))
    pred_cmd.extend(fmt_stat_args("y", y_stats))
    run(pred_cmd)

    pred = json.loads(pred_json.read_text(encoding="utf-8"))
    pred_map = pred["pred_rel_err"]
    actual_map = actual["actual_errors"]

    merged: dict[str, dict[str, float]] = {}
    for name, avals in actual_map.items():
        pred_val = pred_map.get(name)
        merged[name] = {
            "pred_rel_err": float(pred_val) if pred_val is not None else None,
            "actual_sum_rel_err": float(avals["sum_rel_err"]),
            "actual_l2_rel_err": float(avals["l2_rel_err"]),
        }

    compare_json.write_text(
        json.dumps(
            {
                "benchmark": "atax",
                "ir": str(args.ir),
                "rows": args.rows,
                "cols": args.cols,
                "dist_type": args.dist_type,
                "seed": args.seed,
                "results": merged,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with open(compare_csv, "w", encoding="utf-8") as f:
        f.write("target,pred_rel_err,actual_sum_rel_err,actual_l2_rel_err\n")
        for name, vals in merged.items():
            f.write(
                f"{name},{vals['pred_rel_err']},{vals['actual_sum_rel_err']},{vals['actual_l2_rel_err']}\n"
            )

    print("saved", actual_json)
    print("saved", format_json)
    print("saved", pred_json)
    print("saved", compare_json)
    print("saved", compare_csv)


if __name__ == "__main__":
    main()
