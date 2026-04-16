#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_FORMATS = [
    "posit_8_0",
    "posit_8_1",
    "posit_8_2",
    "posit_16_0",
    "posit_16_1",
    "posit_16_2",
    "posit_32_0",
    "posit_32_1",
    "posit_32_2",
]

ENTITY_FEATURE_COLUMNS = [
    "function",
    "entity",
    "role",
    "bb",
    "def_opcode",
    "use_count",
    "is_loop_carried",
    "is_output_related",
    "fanout",
    "reduction_depth",
    "distance_to_output",
    "same_loop_neighbor_count",
    "num_arith_users",
    "raw",
]

OUTPUT_COLUMNS = [
    "app",
    "family",
    "variant_id",
    "sample_id",
    "source_ir",
    "function",
    "entity_id",
    "role",
    "bb",
    "def_opcode",
    "use_count",
    "is_loop_carried",
    "is_output_related",
    "fanout",
    "reduction_depth",
    "distance_to_output",
    "same_loop_neighbor_count",
    "num_arith_users",
    "format",
    "format_n",
    "format_es",
    "tol_whole_app_rel_err",
    "actual_whole_app_rel_err",
    "is_feasible_under_tol",
    "margin_to_tol",
    "label_status",
    "notes",
]


def parse_format_name(fmt: str) -> tuple[int, int]:
    parts = fmt.split("_")
    if len(parts) != 3 or parts[0] != "posit":
        raise ValueError(f"unsupported format name: {fmt}")
    return int(parts[1]), int(parts[2])


def infer_family_variant(app: str) -> tuple[str, str]:
    if "_v" in app:
        family, variant = app.rsplit("_v", 1)
        return family, f"v{variant}"
    return app, ""


def load_entity_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    missing = [c for c in ENTITY_FEATURE_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise SystemExit(f"entity feature csv missing columns: {missing}")
    return rows


def build_label_rows(
    entity_rows: list[dict[str, str]],
    *,
    app: str,
    source_ir: str,
    tol: float,
    role_filter: str,
    formats: list[str],
) -> list[dict[str, str]]:
    family, variant_id = infer_family_variant(app)
    out: list[dict[str, str]] = []
    for row in entity_rows:
        if row.get("role", "") != role_filter:
            continue
        entity_id = row.get("entity", "")
        for fmt in formats:
            n, es = parse_format_name(fmt)
            out.append(
                {
                    "app": app,
                    "family": family,
                    "variant_id": variant_id,
                    "sample_id": "",
                    "source_ir": source_ir,
                    "function": row.get("function", ""),
                    "entity_id": entity_id,
                    "role": row.get("role", ""),
                    "bb": row.get("bb", ""),
                    "def_opcode": row.get("def_opcode", ""),
                    "use_count": row.get("use_count", ""),
                    "is_loop_carried": row.get("is_loop_carried", ""),
                    "is_output_related": row.get("is_output_related", ""),
                    "fanout": row.get("fanout", ""),
                    "reduction_depth": row.get("reduction_depth", ""),
                    "distance_to_output": row.get("distance_to_output", ""),
                    "same_loop_neighbor_count": row.get("same_loop_neighbor_count", ""),
                    "num_arith_users": row.get("num_arith_users", ""),
                    "format": fmt,
                    "format_n": str(n),
                    "format_es": str(es),
                    "tol_whole_app_rel_err": f"{tol:.6e}",
                    "actual_whole_app_rel_err": "",
                    "is_feasible_under_tol": "",
                    "margin_to_tol": "",
                    "label_status": "pending_eval",
                    "notes": "accumulator-only skeleton; fill actual_whole_app_rel_err / is_feasible_under_tol / margin_to_tol after single-entity substitution run",
                }
            )
    return out


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("entity_csv", help="entity feature csv from extract_entity_features.py")
    ap.add_argument("--out", required=True, help="output entity_label_dataset.csv")
    ap.add_argument("--app", default=None, help="app name; default is entity_csv stem without suffixes")
    ap.add_argument("--source-ir", default="", help="source LLVM IR path recorded into the dataset")
    ap.add_argument("--tol", type=float, default=1e-3, help="binary label threshold on whole-app rel err")
    ap.add_argument(
        "--role",
        default="accumulator",
        choices=["accumulator", "loop_carried_state", "reduction_intermediate"],
        help="entity role to expand into label rows; default accumulator",
    )
    ap.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help="comma-separated posit formats, default all posit(8/16/32,0..2)",
    )
    args = ap.parse_args()

    entity_csv = Path(args.entity_csv)
    if not entity_csv.exists():
        raise SystemExit(f"entity csv not found: {entity_csv}")

    app = args.app or entity_csv.stem.replace("_entity_features", "")
    formats = [x.strip() for x in str(args.formats).split(",") if x.strip()]

    rows = load_entity_rows(entity_csv)
    out_rows = build_label_rows(
        rows,
        app=app,
        source_ir=str(args.source_ir),
        tol=float(args.tol),
        role_filter=str(args.role),
        formats=formats,
    )
    write_rows(Path(args.out), out_rows)
    print(f"saved {args.out}")
    print(f"app={app}")
    print(f"role={args.role}")
    print(f"entity_rows={sum(1 for r in rows if r.get('role', '') == args.role)}")
    print(f"label_rows={len(out_rows)}")


if __name__ == "__main__":
    main()
