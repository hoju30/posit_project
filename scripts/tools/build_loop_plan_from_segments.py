#!/usr/bin/env python3
"""
Build an onnx-mlir loop plan.json by mapping LLVM loop-segments -> MLIR loop_id.

Why this exists
---------------
In posit_test we can split an LLVM IR (.ll) into loop-extracted segments and run the
regression model per segment. In onnx-mlir, we attach a plan keyed by MLIR loop ids
(posit.loop_id) on scf.for / krnl.iterate ops.

This script bridges the two worlds by *pairing segments to loop_ids by order*:
  - segment order: predictions_json["segments"] sorted by segment_id
  - loop order: posit.loop_id appearance order in MLIR after --posit-assign-loop-id

Important assumption
--------------------
This "by-index" mapping assumes the loop order (and count) in the MLIR you feed
to the compiler matches the loop extraction order from the LLVM IR you analyzed.
For demo kernels (e.g. dot) this is often true (sometimes trivially: only 1 loop),
but for real programs this may break if lowering introduces/reorders loops.

If you need a robust mapping, you must propagate a stable loop identifier across
lowering (e.g., attach IDs early and preserve them down to LLVM).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


LOOP_ID_RE = re.compile(r'posit\.loop_id\s*=\s*"([^"]+)"')


def load_predictions(pred_json: Path) -> list[dict[str, Any]]:
    obj = json.loads(pred_json.read_text(encoding="utf-8"))
    segs = obj.get("segments", [])
    if not isinstance(segs, list) or not segs:
        raise SystemExit("predictions json must contain non-empty array: segments")
    # stable: by segment_id, fallback to original order
    def key(seg: dict[str, Any]) -> int:
        try:
            return int(seg.get("segment_id", 1 << 30))
        except Exception:
            return 1 << 30

    segs_sorted = sorted([s for s in segs if isinstance(s, dict)], key=key)
    return segs_sorted


def extract_loop_ids(onnx_mlir_opt: Path, mlir_path: Path) -> list[str]:
    cmd = [str(onnx_mlir_opt), str(mlir_path), "--posit-assign-loop-id"]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    loop_ids = LOOP_ID_RE.findall(res.stdout)
    # preserve order; dedup just in case an op is printed twice
    out: list[str] = []
    seen = set()
    for lid in loop_ids:
        if lid in seen:
            continue
        seen.add(lid)
        out.append(lid)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-json", required=True, help="from scripts/predict_ir_errors_by_loops.py --save-json")
    p.add_argument("--mlir", required=True, help="MLIR file to extract loop_ids from (via --posit-assign-loop-id)")
    p.add_argument(
        "--onnx-mlir-opt",
        default="/home/hoju/test/onnx-mlir/build/Release/bin/onnx-mlir-opt",
        help="path to onnx-mlir-opt binary",
    )
    p.add_argument("--out", required=True, help="output plan.json path (loop_id-keyed)")
    p.add_argument(
        "--include-chosen",
        action="store_true",
        help="include chosen/chosen_pred from segment predictions (otherwise omit and let pass infer argmin)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="fail if loop count != segment count (otherwise map min(counts) and warn)",
    )
    args = p.parse_args()

    pred_json = Path(args.predictions_json)
    mlir_path = Path(args.mlir)
    opt_path = Path(args.onnx_mlir_opt)
    out_path = Path(args.out)

    if not pred_json.exists():
        raise SystemExit(f"predictions json not found: {pred_json}")
    if not mlir_path.exists():
        raise SystemExit(f"mlir not found: {mlir_path}")
    if not opt_path.exists():
        raise SystemExit(f"onnx-mlir-opt not found: {opt_path}")

    segs = load_predictions(pred_json)
    loop_ids = extract_loop_ids(opt_path, mlir_path)

    if not loop_ids:
        raise SystemExit("no posit.loop_id found in MLIR; is it too early/late in the pipeline?")

    if len(loop_ids) != len(segs):
        msg = f"loop count ({len(loop_ids)}) != segment count ({len(segs)})"
        if args.strict:
            raise SystemExit(msg)
        print("[warn]", msg)

    n = min(len(loop_ids), len(segs))
    loops_obj: dict[str, Any] = {}
    for i in range(n):
        lid = loop_ids[i]
        seg = segs[i]
        entry: dict[str, Any] = {
            "pred_rel_err": dict(seg.get("pred_rel_err", {})),
            # debug payload (pass will ignore unknown keys)
            "_segment": {
                "segment_id": seg.get("segment_id", None),
                "func_name": seg.get("func_name", ""),
                "path": seg.get("path", ""),
            },
        }
        if args.include_chosen:
            if "chosen" in seg:
                entry["chosen"] = seg["chosen"]
            if "chosen_pred" in seg:
                entry["chosen_pred"] = seg["chosen_pred"]
        loops_obj[str(lid)] = entry

    plan = {"loops": loops_obj}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[ok] loops:", len(loop_ids), "segments:", len(segs), "mapped:", n)
    print("[ok] wrote:", str(out_path))
    print("Next:")
    print(
        f'  {opt_path} {mlir_path} --posit-assign-loop-id '
        f'--posit-attach-plan="posit-plan-json={out_path.resolve()}" > out.mlir'
    )


if __name__ == "__main__":
    main()

