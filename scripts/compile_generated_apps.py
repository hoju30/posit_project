#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default=str(ROOT / "generated_apps" / "apps_manifest.csv"),
        help="source app manifest csv from generate_source_apps.py",
    )
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "generated_apps" / "compiled"),
        help="root output dir for llvm ir / entity csv / label csv",
    )
    ap.add_argument("--clang", default="clang-16")
    ap.add_argument("--opt-level", default="O1", choices=["O0", "O1", "O2", "O3"])
    ap.add_argument("--role", default="accumulator", choices=["accumulator", "loop_carried_state", "reduction_intermediate"])
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")

    out_dir = Path(args.out_dir)
    ir_dir = out_dir / "ir"
    feat_dir = out_dir / "entity_features"
    label_dir = out_dir / "entity_labels"
    ir_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str]] = []
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        app = row["app"]
        family = row["family"]
        variant_id = row["variant_id"]
        src = Path(row["source_path"])
        if not src.exists():
            continue

        ll_path = ir_dir / family / f"{app}.ll"
        feat_csv = feat_dir / family / f"{app}_entity_features.csv"
        feat_json = feat_dir / family / f"{app}_entity_features.json"
        label_csv = label_dir / family / f"{app}_entity_label_dataset.csv"

        ll_path.parent.mkdir(parents=True, exist_ok=True)
        feat_csv.parent.mkdir(parents=True, exist_ok=True)
        label_csv.parent.mkdir(parents=True, exist_ok=True)

        run([args.clang, "-S", "-emit-llvm", f"-{args.opt_level}", str(src), "-o", str(ll_path)])
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "extract_entity_features.py"),
                str(ll_path),
                "--out-json",
                str(feat_json),
                "--out-csv",
                str(feat_csv),
            ]
        )
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "generate_entity_labels.py"),
                str(feat_csv),
                "--app",
                app,
                "--source-ir",
                str(ll_path),
                "--role",
                str(args.role),
                "--tol",
                str(args.tol),
                "--out",
                str(label_csv),
            ]
        )

        entity_rows = 0
        with feat_csv.open(newline="", encoding="utf-8") as ef:
            ereader = csv.DictReader(ef)
            for erow in ereader:
                if erow.get("role", "") == args.role:
                    entity_rows += 1
        label_rows = 0
        with label_csv.open(newline="", encoding="utf-8") as lf:
            lreader = csv.DictReader(lf)
            for _ in lreader:
                label_rows += 1

        summary_rows.append(
            {
                "app": app,
                "family": family,
                "variant_id": variant_id,
                "source_path": str(src.resolve()),
                "ir_path": str(ll_path.resolve()),
                "entity_feature_csv": str(feat_csv.resolve()),
                "entity_label_csv": str(label_csv.resolve()),
                "role": str(args.role),
                "entity_rows": str(entity_rows),
                "label_rows": str(label_rows),
            }
        )

    summary_csv = out_dir / "compiled_manifest.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "app",
                "family",
                "variant_id",
                "source_path",
                "ir_path",
                "entity_feature_csv",
                "entity_label_csv",
                "role",
                "entity_rows",
                "label_rows",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"processed {len(summary_rows)} apps")
    print(f"saved {summary_csv}")


if __name__ == "__main__":
    main()
