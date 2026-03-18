#!/usr/bin/env python3
"""
Generate a loop-id keyed plan.json for onnx-mlir passes (external tool workflow).

This script:
  1) Runs onnx-mlir-opt with --posit-assign-loop-id on an input MLIR file
  2) Extracts all generated posit.loop_id values
  3) Writes a plan JSON in the format expected by --posit-attach-plan

Note:
  - This is a smoke-test / wiring tool. It does NOT run the regression model yet.
  - You can start with a fixed chosen format for all loops, then later upgrade this
    script to call the predictor per loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path


LOOP_ID_RE = re.compile(r'posit\.loop_id\s*=\s*"([^"]+)"')


def unique_preserve(xs: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mlir", help="input MLIR file")
    p.add_argument(
        "--onnx-mlir-opt",
        default="/home/hoju/test/onnx-mlir/build/Release/bin/onnx-mlir-opt",
        help="path to onnx-mlir-opt binary",
    )
    p.add_argument(
        "--out",
        default="plan.json",
        help="output plan JSON path",
    )
    p.add_argument(
        "--save-annotated-mlir",
        default=None,
        help="optional path to save MLIR after --posit-assign-loop-id",
    )
    p.add_argument(
        "--pred-csv",
        default=None,
        help="optional predictions CSV produced by scripts/predict_ir_errors.py --save-csv "
        "(columns: target,pred). If provided, fills pred_rel_err for every loop.",
    )
    p.add_argument(
        "--pred-json",
        default=None,
        help="optional predictions JSON produced by scripts/predict_ir_errors.py --save-json "
        "(expects key: pred_rel_err). If provided, fills pred_rel_err for every loop.",
    )
    p.add_argument(
        "--chosen",
        default="",
        help="optional chosen format for all loops (e.g. posit_16_1, fp32). "
        "If empty, omit 'chosen' and let the compiler pass infer argmin(pred_rel_err).",
    )
    p.add_argument(
        "--chosen-pred",
        type=float,
        default=0.0,
        help="optional chosen_pred value to store (float)",
    )
    args = p.parse_args()

    in_path = Path(args.mlir)
    if not in_path.exists():
        raise SystemExit(f"MLIR not found: {in_path}")

    opt_path = Path(args.onnx_mlir_opt)
    if not opt_path.exists():
        raise SystemExit(f"onnx-mlir-opt not found: {opt_path}")

    cmd = [str(opt_path), str(in_path), "--posit-assign-loop-id"]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    annotated = res.stdout

    if args.save_annotated_mlir:
        out_mlir = Path(args.save_annotated_mlir)
        out_mlir.parent.mkdir(parents=True, exist_ok=True)
        out_mlir.write_text(annotated, encoding="utf-8")

    loop_ids = unique_preserve(LOOP_ID_RE.findall(annotated))
    if not loop_ids:
        raise SystemExit(
            "No posit.loop_id found. Make sure the MLIR contains scf.for/krnl.iterate, "
            "and that --posit-assign-loop-id runs at a stage where loops exist."
        )

    if args.pred_csv and args.pred_json:
        raise SystemExit("use only one of --pred-csv or --pred-json")

    pred_map: dict[str, float] = {}
    if args.pred_csv:
        pred_path = Path(args.pred_csv)
        if not pred_path.exists():
            raise SystemExit(f"pred csv not found: {pred_path}")
        with pred_path.open("r", newline="") as f:
            r = csv.DictReader(f)
            if not r.fieldnames or "target" not in r.fieldnames or "pred" not in r.fieldnames:
                raise SystemExit("pred csv must have header: target,pred")
            for row in r:
                t = (row.get("target") or "").strip()
                v = (row.get("pred") or "").strip()
                if not t or not v:
                    continue
                try:
                    pred_map[t] = float(v)
                except Exception:
                    continue
        if not pred_map:
            raise SystemExit(f"pred csv has no valid rows: {pred_path}")

    if args.pred_json:
        pred_path = Path(args.pred_json)
        if not pred_path.exists():
            raise SystemExit(f"pred json not found: {pred_path}")
        obj = json.loads(pred_path.read_text(encoding="utf-8"))
        preds = obj.get("pred_rel_err")
        if not isinstance(preds, dict) or not preds:
            raise SystemExit("pred json must contain non-empty object: pred_rel_err")
        for k, v in preds.items():
            if not isinstance(k, str):
                continue
            try:
                pred_map[k] = float(v)
            except Exception:
                continue
        if not pred_map:
            raise SystemExit("pred json pred_rel_err has no valid numeric values")

    plan = {
        "loops": {
            lid: {
                **({"chosen": str(args.chosen)} if str(args.chosen) else {}),
                **({"chosen_pred": float(args.chosen_pred)} if str(args.chosen) else {}),
                "pred_rel_err": dict(pred_map),
            }
            for lid in loop_ids
        }
    }

    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[ok] loops:", len(loop_ids))
    print("[ok] wrote:", str(out_json))
    if args.save_annotated_mlir:
        print("[ok] annotated mlir:", str(Path(args.save_annotated_mlir)))
    print("Next:")
    print(
        f'  {opt_path} {in_path} --posit-assign-loop-id '
        f'--posit-attach-plan="posit-plan-json={out_json.resolve()}" > out.mlir'
    )


if __name__ == "__main__":
    main()
