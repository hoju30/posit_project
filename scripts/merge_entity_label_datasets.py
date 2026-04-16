#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--label-status",
        default="actual_eval",
        help="only keep rows with this label_status; default actual_eval",
    )
    args = ap.parse_args()

    manifest = Path(args.compiled_manifest)
    if not manifest.exists():
        raise SystemExit(f"compiled manifest not found: {manifest}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, str]] = []
    fieldnames: list[str] | None = None

    with manifest.open(newline="", encoding="utf-8") as f:
        for mrow in csv.DictReader(f):
            label_csv = Path(mrow["entity_label_csv"])
            if not label_csv.exists():
                continue
            with label_csv.open(newline="", encoding="utf-8") as lf:
                reader = csv.DictReader(lf)
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames or [])
                for row in reader:
                    if row.get("label_status") != args.label_status:
                        continue
                    rows_out.append(row)

    if fieldnames is None:
        raise SystemExit("no entity label rows found")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    print(f"saved {out_path}")
    print(f"rows={len(rows_out)}")


if __name__ == "__main__":
    main()
