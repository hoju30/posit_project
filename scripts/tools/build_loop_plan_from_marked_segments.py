#!/usr/bin/env python3
"""
Build onnx-mlir loop plan.json by matching LLVM loop-segments to MLIR loop_id via marker calls.

Prereqs
-------
1) In MLIR, run:
     --posit-assign-loop-id
     --posit-insert-loop-marker

   This inserts a call:
     call @__posit_loop_marker(%hash)
   into each loop body, where %hash = FNV1a64(loop_id_string).

2) Lower MLIR -> LLVM IR, then run loop extraction on the LLVM IR.
   Each extracted segment should still contain the marker call, so we can recover
   which MLIR loop it came from.

3) Run scripts/predict_ir_errors_by_loops.py on the extracted LLVM segments to get:
     predictions.json (contains segments[].pred_rel_err and segments[].path)

This script then:
  - extracts loop_ids from the MLIR (by invoking onnx-mlir-opt --posit-assign-loop-id)
  - computes FNV1a64(loop_id) for each loop_id
  - parses each LLVM segment .ll file to find the marker hash
  - joins segment predictions onto loop_ids
  - aggregates multiple segments per loop_id (default: max per target)
  - writes plan.json keyed by loop_id, suitable for --posit-attach-plan
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


LOOP_ID_RE = re.compile(r'posit\.loop_id\s*=\s*"([^"]+)"')


def fnv1a64(s: str) -> int:
    h = 14695981039346656037
    prime = 1099511628211
    for b in s.encode("utf-8", errors="strict"):
        h ^= b
        h = (h * prime) & 0xFFFFFFFFFFFFFFFF
    return h


def extract_loop_ids(onnx_mlir_opt: Path, mlir_path: Path) -> list[str]:
    cmd = [str(onnx_mlir_opt), str(mlir_path), "--posit-assign-loop-id"]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    loop_ids = LOOP_ID_RE.findall(res.stdout)
    out: list[str] = []
    seen = set()
    for lid in loop_ids:
        if lid in seen:
            continue
        seen.add(lid)
        out.append(lid)
    return out


def sanitize_for_symbol(s: str) -> str:
    out_chars: list[str] = []
    for ch in s:
        if ch.isalnum() or ch == "_":
            out_chars.append(ch)
        else:
            out_chars.append("_")
    out = "".join(out_chars)
    return out if out else "loop"


def build_sanitized_loop_id_map(loop_ids: list[str]) -> dict[str, str]:
    m: dict[str, str] = {}
    for lid in loop_ids:
        key = sanitize_for_symbol(lid)
        # Keep first seen on collision; collisions are rare for our loop_id pattern.
        if key not in m:
            m[key] = lid
    return m


def map_loop_id_from_segment_func_name(
    func_name: str, sanitized_map: dict[str, str], outlined_prefix: str = "__posit_seg_"
) -> str | None:
    if not func_name.startswith(outlined_prefix):
        return None
    key = func_name[len(outlined_prefix) :]
    if key in sanitized_map:
        return sanitized_map[key]
    # Outline pass may append hash/salt suffix on collisions:
    #   <sanitized_loop_id>_h<hex>[_<salt>]
    key2 = re.sub(r"_h[0-9a-fA-F]+(?:_[0-9]+)?$", "", key)
    return sanitized_map.get(key2)


def load_segments(predictions_json: Path) -> list[dict[str, Any]]:
    obj = json.loads(predictions_json.read_text(encoding="utf-8"))
    segs = obj.get("segments", [])
    if not isinstance(segs, list) or not segs:
        raise SystemExit("predictions json must contain non-empty array: segments")
    return [s for s in segs if isinstance(s, dict)]


def parse_marker_hash(ll_path: Path, marker_func: str) -> int | None:
    # Matches (typical):
    #   call void @__posit_loop_marker(i64 123)
    #   call void @__posit_loop_marker(i64 %0)   (won't match)
    # We only support immediate integer constants (what the MLIR pass emits).
    pat = re.compile(rf"@{re.escape(marker_func)}\s*\(\s*i64\s+(-?[0-9]+)\s*\)")
    text = ll_path.read_text(encoding="utf-8", errors="ignore")
    hits = pat.findall(text)
    if not hits:
        return None
    try:
        vals = [(int(x) & 0xFFFFFFFFFFFFFFFF) for x in hits]
        uniq = sorted(set(vals))
        # If a single segment contains multiple loop markers, mapping is ambiguous.
        if len(uniq) != 1:
            return None
        return uniq[0]
    except Exception:
        return None


def agg_pred_dicts(pred_dicts: list[dict[str, float]], *, agg: str) -> dict[str, float]:
    if not pred_dicts:
        return {}
    keys = sorted({k for d in pred_dicts for k in d.keys()})
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(d[k]) for d in pred_dicts if k in d]
        if not vals:
            continue
        if agg == "mean":
            out[k] = sum(vals) / float(len(vals))
        else:
            out[k] = max(vals)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-json", required=True, help="from predict_ir_errors_by_loops.py --save-json")
    p.add_argument("--mlir", required=True, help="MLIR file (loop_ids extracted via --posit-assign-loop-id)")
    p.add_argument(
        "--onnx-mlir-opt",
        default="/home/hoju/test/onnx-mlir/build/Release/bin/onnx-mlir-opt",
        help="path to onnx-mlir-opt",
    )
    p.add_argument("--out", required=True, help="output plan.json")
    p.add_argument("--marker-func", default="__posit_loop_marker", help="marker function name")
    p.add_argument("--agg", default="max", choices=["max", "mean"], help="aggregation for multi-segment per loop")
    p.add_argument(
        "--strict",
        action="store_true",
        help="fail if any segment cannot be mapped to a loop_id",
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

    loop_ids = extract_loop_ids(opt_path, mlir_path)
    if not loop_ids:
        raise SystemExit("no posit.loop_id found in MLIR; is it too early/late in the pipeline?")

    hash_to_loop_id: dict[int, str] = {}
    for lid in loop_ids:
        h = fnv1a64(lid)
        hash_to_loop_id[h] = lid
    sanitized_loop_id_map = build_sanitized_loop_id_map(loop_ids)

    segs = load_segments(pred_json)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unmapped: list[str] = []

    for seg in segs:
        fn = str(seg.get("func_name", ""))
        lid_by_name = map_loop_id_from_segment_func_name(fn, sanitized_loop_id_map)
        if lid_by_name is not None:
            buckets[lid_by_name].append(seg)
            continue

        pth = Path(str(seg.get("path", "")))
        if not pth.exists():
            unmapped.append(str(pth))
            continue
        h = parse_marker_hash(pth, str(args.marker_func))
        if h is None:
            unmapped.append(str(pth))
            continue
        lid = hash_to_loop_id.get(h)
        if lid is None:
            unmapped.append(str(pth))
            continue
        buckets[lid].append(seg)

    if unmapped:
        msg = f"unmapped segments: {len(unmapped)}"
        if args.strict:
            raise SystemExit(msg + " (first: " + unmapped[0] + ")")
        print("[warn]", msg, "(first:", unmapped[0], ")")

    loops_obj: dict[str, Any] = {}
    for lid, seg_list in buckets.items():
        pred_dicts: list[dict[str, float]] = []
        for s in seg_list:
            pd = s.get("pred_rel_err", {})
            if isinstance(pd, dict):
                pred_dicts.append({str(k): float(v) for k, v in pd.items()})
        pred_agg = agg_pred_dicts(pred_dicts, agg=str(args.agg))
        loops_obj[lid] = {
            "pred_rel_err": pred_agg,
            "_segments": [
                {
                    "segment_id": s.get("segment_id", None),
                    "func_name": s.get("func_name", ""),
                    "path": s.get("path", ""),
                }
                for s in seg_list
            ],
        }

    plan = {"loops": loops_obj}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[ok] loop_ids:", len(loop_ids))
    print("[ok] mapped loops:", len(loops_obj))
    print("[ok] wrote:", str(out_path))
    print("Next:")
    print(
        f'  {opt_path} {mlir_path} --posit-assign-loop-id '
        f'--posit-attach-plan="posit-plan-json={out_path.resolve()}" > out.mlir'
    )


if __name__ == "__main__":
    main()
