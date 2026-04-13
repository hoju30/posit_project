#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "benchmark_bicg_eval.cpp"
BIN = ROOT / "bin" / "bicg_eval"
IR = Path("/home/hoju/test/polybench-c-3.2/bicg.ll")
POLYBENCH_ROOT = Path("/home/hoju/test/polybench-c-3.2")


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


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


def emit_ir(ir_path: Path) -> None:
    run(
        [
            "clang-16",
            "-S",
            "-emit-llvm",
            "-O2",
            "-I",
            "utilities",
            "-I",
            "linear-algebra/kernels/bicg",
            "linear-algebra/kernels/bicg/bicg.c",
            "-o",
            str(ir_path),
        ],
        cwd=POLYBENCH_ROOT,
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
    ap.add_argument("--formats", default="", help="optional subset, e.g. 16:1,16:2,32:0,32:1,32:2")
    ap.add_argument("--model", default=str(ROOT / "models" / "ir2vec_error_predictor.joblib"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "bicg_benchmark_eval"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actual_json = out_dir / "actual.json"
    pred_json = out_dir / "pred.json"
    format_json = out_dir / "format_features.json"
    compare_json = out_dir / "compare.json"
    compare_csv = out_dir / "compare.csv"

    compile_binary()
    emit_ir(IR)

    eval_cmd = [
        str(BIN),
        "--rows",
        str(args.rows),
        "--cols",
        str(args.cols),
        "--polybench-init",
        "--out-json",
        str(actual_json),
    ]
    if args.formats:
        eval_cmd.extend(["--formats", args.formats])
    run(eval_cmd)

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
        str(IR),
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
                "benchmark": "bicg",
                "ir": str(IR),
                "rows": args.rows,
                "cols": args.cols,
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
