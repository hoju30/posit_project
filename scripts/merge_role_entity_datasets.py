#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUTS = [
    ROOT / "data" / "generated_entity_label_dataset_accum.csv",
    ROOT / "data" / "generated_entity_label_dataset_lcs.csv",
    ROOT / "data" / "generated_entity_label_dataset_ri.csv",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[str(p) for p in DEFAULT_INPUTS],
        help="entity-level csv datasets to merge",
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "data" / "generated_entity_label_dataset_all.csv"),
        help="merged output csv path",
    )
    ap.add_argument(
        "--label-status",
        default="actual_eval",
        help="only keep rows with this label_status; default actual_eval",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] | None = None
    rows_out: list[dict[str, str]] = []

    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        if not input_path.exists():
            raise SystemExit(f"dataset not found: {input_path}")
        with input_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                if row.get("label_status") != args.label_status:
                    continue
                rows_out.append(row)

    if fieldnames is None:
        raise SystemExit("no input rows found")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    print(f"saved {out_path}")
    print(f"rows={len(rows_out)}")


if __name__ == "__main__":
    main()
