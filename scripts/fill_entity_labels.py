#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ACCUM_SRC = ROOT / "src" / "generated_app_accumulator_eval.cpp"
ACCUM_BIN = ROOT / "bin" / "generated_app_accumulator_eval"
RED_TMP_SRC = ROOT / "src" / "generated_app_reduction_intermediate_eval.cpp"
RED_TMP_BIN = ROOT / "bin" / "generated_app_reduction_intermediate_eval"
DEFAULT_FORMATS = "8:0,8:1,8:2,16:0,16:1,16:2,32:0,32:1,32:2"
SUPPORTED_FAMILIES_BY_ROLE = {
    "accumulator": {
        "sum_reduce",
        "dot_product",
        "weighted_sum",
        "prefix_recurrence",
        "two_stage_reduce",
        "stencil1d_like",
    },
    "loop_carried_state": {
        "ema_recurrence",
    },
    "reduction_intermediate": {
        "dot_product",
        "weighted_sum",
        "prefix_recurrence",
        "ema_recurrence",
        "two_stage_reduce",
        "stencil1d_like",
    },
}


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def compile_binary(src: Path, out_bin: Path) -> None:
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "g++",
            "-std=c++20",
            "-O2",
            str(src),
            "-I" + str(ROOT / "external" / "universal" / "include" / "sw"),
            "-o",
            str(out_bin),
        ]
    )


def infer_variant_int(variant_id: str) -> int:
    text = variant_id.strip()
    if text.startswith("v"):
        text = text[1:]
    return int(text or "0")


def load_target_entity_ids(entity_feature_csv: Path, role: str) -> list[str]:
    with entity_feature_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if role == "accumulator":
        return [row["entity"] for row in rows if row.get("role") == "accumulator"]
    if role == "loop_carried_state":
        out = []
        for row in rows:
            if row.get("role") != "loop_carried_state":
                continue
            raw = row.get("raw", "")
            if "phi double" in raw or "phi float" in raw:
                out.append(row["entity"])
        return out
    if role == "reduction_intermediate":
        candidates = []
        for row in rows:
            if row.get("role") != "reduction_intermediate":
                continue
            opcode = row.get("def_opcode", "")
            if opcode not in {"fadd", "fsub", "fmul", "fdiv", "fmuladd", "fma"}:
                continue
            try:
                score = (
                    int(row.get("is_output_related", "0")),
                    int(float(row.get("num_arith_users", "0"))),
                    int(float(row.get("use_count", "0"))),
                    int(float(row.get("reduction_depth", "0"))),
                    int(float(row.get("fanout", "0"))),
                    -int(float(row.get("distance_to_output", "999999"))),
                )
            except ValueError:
                continue
            candidates.append((score, row["entity"]))
        if not candidates:
            return []
        candidates.sort(reverse=True)
        return [candidates[0][1]]
    return []


def fill_label_csv(
    label_csv: Path,
    eval_json: Path,
    *,
    tol: float,
    overwrite: bool,
    role: str,
    target_entity_ids: set[str],
) -> tuple[int, int]:
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    results = data["results"]
    with label_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    updated = 0
    skipped = 0
    for row in rows:
        if row.get("role") != role:
            skipped += 1
            continue
        if row.get("entity_id", "") not in target_entity_ids:
            row["label_status"] = "skipped_non_target_entity"
            row["notes"] = "non-floating or unsupported entity for this role-specific evaluator"
            skipped += 1
            continue
        if row.get("label_status") == "actual_eval" and not overwrite:
            skipped += 1
            continue
        fmt = row.get("format", "")
        actual = results.get(fmt)
        if actual is None:
            row["label_status"] = "missing_format_eval"
            skipped += 1
            continue
        err = float(actual["actual_whole_app_rel_err"])
        row["actual_whole_app_rel_err"] = f"{err:.6e}"
        row["is_feasible_under_tol"] = "1" if err <= tol else "0"
        row["margin_to_tol"] = f"{(tol - err):.6e}"
        row["label_status"] = "actual_eval"
        row["notes"] = f"single-{role} substitution; other entities fixed to fp64"
        updated += 1

    with label_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return updated, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--compiled-manifest",
        default=str(ROOT / "generated_apps" / "compiled" / "compiled_manifest.csv"),
        help="compiled manifest from compile_generated_apps.py",
    )
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "generated_apps" / "compiled" / "actual_eval"),
        help="directory for per-app eval json files",
    )
    ap.add_argument("--formats", default=DEFAULT_FORMATS)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--role", default="accumulator", choices=["accumulator", "loop_carried_state", "reduction_intermediate"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="optional app limit for smoke tests")
    args = ap.parse_args()

    manifest = Path(args.compiled_manifest)
    if not manifest.exists():
        raise SystemExit(f"compiled manifest not found: {manifest}")

    role = str(args.role)
    if role == "accumulator" or role == "loop_carried_state":
        evaluator_bin = ACCUM_BIN
        compile_binary(ACCUM_SRC, evaluator_bin)
    elif role == "reduction_intermediate":
        evaluator_bin = RED_TMP_BIN
        compile_binary(RED_TMP_SRC, evaluator_bin)
    else:
        raise SystemExit(f"unsupported role: {role}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    processed = 0
    updated_rows = 0
    skipped_apps = 0

    for row in rows:
        if args.limit and processed >= args.limit:
            break
        if row.get("role") != role:
            skipped_apps += 1
            continue
        family = row.get("family", "")
        if family not in SUPPORTED_FAMILIES_BY_ROLE[role]:
            skipped_apps += 1
            continue
        entity_feature_csv = Path(row["entity_feature_csv"])
        target_entity_ids = load_target_entity_ids(entity_feature_csv, role)
        if len(target_entity_ids) != 1:
            skipped_apps += 1
            continue

        app = row["app"]
        variant = infer_variant_int(row.get("variant_id", ""))
        label_csv = Path(row["entity_label_csv"])
        if not label_csv.exists():
            skipped_apps += 1
            continue

        eval_json = out_dir / family / f"{app}_actual_eval.json"
        eval_json.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                str(evaluator_bin),
                "--family",
                family,
                "--variant",
                str(variant),
                "--formats",
                str(args.formats),
                "--out-json",
                str(eval_json),
            ]
        )
        updated, _ = fill_label_csv(
            label_csv,
            eval_json,
            tol=float(args.tol),
            overwrite=bool(args.overwrite),
            role=role,
            target_entity_ids=set(target_entity_ids),
        )
        updated_rows += updated
        processed += 1

    print(f"processed_apps={processed}")
    print(f"updated_label_rows={updated_rows}")
    print(f"skipped_apps={skipped_apps}")
    print(f"eval_json_dir={out_dir}")


if __name__ == "__main__":
    main()
