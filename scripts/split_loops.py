#!/usr/bin/env python3
"""
Split an input LLVM IR into per-loop segments.

Current role:
  - keep only loop segmentation
  - no longer performs CFG / SCC analysis

This module is imported by other scripts for:
  - split_ir_by_loops(...)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent

FUNC_DEF_RE = re.compile(r"^define\s+.*@([^(]+)\(", re.IGNORECASE)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def parse_defined_functions(ir_text: str) -> list[str]:
    names: list[str] = []
    for line in ir_text.splitlines():
        m = FUNC_DEF_RE.match(line.strip())
        if m:
            names.append(m.group(1))
    return names


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def split_ir_by_loops(
    input_ir: str | Path,
    *,
    out_dir: str | Path,
    force: bool = False,
    split_mode: str = "outlined",
    opt: str = "opt-16",
    llvm_extract: str = "llvm-extract-16",
    include_regex: str = r"^__posit_seg_",
    recursive: bool = False,
    allow_no_extract: bool = False,
) -> Path:
    """
    Split an LLVM IR module into per-loop segments.
    - split_mode=outlined: extract functions directly from input IR by include_regex
    - split_mode=loop-extract: run opt loop-extract first, then extract matched funcs

    Returns the manifest.json path under out_dir.
    """
    in_path = Path(input_ir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    manifest = out_dir_p / "manifest.json"
    if manifest.exists() and not force:
        return manifest

    include_re = re.compile(include_regex)
    if split_mode not in {"outlined", "loop-extract"}:
        raise SystemExit(f"unsupported split mode: {split_mode}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        extracted_module = td_path / f"{in_path.stem}.segments_source.ll"
        if split_mode == "loop-extract":
            run(
                [
                    opt,
                    "-enable-new-pm=0",
                    "-S",
                    "-loop-simplify",
                    "-loop-extract",
                    str(in_path),
                    "-o",
                    str(extracted_module),
                ]
            )
        else:
            extracted_module.write_text(
                in_path.read_text(encoding="utf-8", errors="ignore"),
                encoding="utf-8",
            )

        ir_text = extracted_module.read_text(encoding="utf-8", errors="ignore")
        funcs = parse_defined_functions(ir_text)
        loop_funcs = [f for f in funcs if include_re.search(f)]
        if not loop_funcs:
            if not allow_no_extract:
                raise SystemExit(
                    "[warn] no matching segment functions found; "
                    f"mode={split_mode}, include_regex={include_regex}"
                )
            if not funcs:
                raise SystemExit("[warn] no functions found in IR for segmentation")
            loop_funcs = [funcs[0]]
            sys.stderr.write(
                "[warn] no segment functions found; fallback to single segment: "
                f"{loop_funcs[0]}\n"
            )

        manifest_obj: dict[str, Any] = {
            "input_ir": str(in_path.resolve()),
            "out_dir": str(out_dir_p.resolve()),
            "split_mode": split_mode,
            "include_regex": include_regex,
            "segments": [],
        }

        for i, fn in enumerate(loop_funcs):
            seg_name = f"{i:03d}_{safe_filename(fn)}.ll"
            seg_path = out_dir_p / seg_name
            cmd = [llvm_extract, "-S", "--func", fn]
            if recursive:
                cmd.append("--recursive")
            cmd += [str(extracted_module), "-o", str(seg_path)]
            run(cmd)
            manifest_obj["segments"].append(
                {
                    "segment_id": i,
                    "func_name": fn,
                    "path": str(seg_path.resolve()),
                }
            )

        manifest.write_text(
            json.dumps(manifest_obj, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ir", help="input LLVM IR (.ll)")
    p.add_argument(
        "--split-mode",
        default="outlined",
        choices=["outlined", "loop-extract"],
        help="segmentation mode: outlined funcs in IR (default) or LLVM loop-extract",
    )
    p.add_argument(
        "--segments-dir",
        default=None,
        help="directory to store/read loop segments (default: ir/segments/<stem>)",
    )
    p.add_argument("--force-split", action="store_true", help="re-run loop splitting even if manifest exists")
    p.add_argument("--opt", default="opt-16", help="opt binary (default: opt-16)")
    p.add_argument("--llvm-extract", default="llvm-extract-16", help="llvm-extract binary (default: llvm-extract-16)")
    p.add_argument(
        "--include-regex",
        default=r"^__posit_seg_",
        help=r"regex for segment function names (default: ^__posit_seg_)",
    )
    p.add_argument("--recursive", action="store_true", help="pass --recursive to llvm-extract")
    p.add_argument(
        "--allow-no-extract",
        action="store_true",
        help="if no segment functions are matched, fall back to a single segment",
    )
    args = p.parse_args()

    in_path = Path(args.ir)
    if not in_path.exists():
        raise SystemExit(f"IR not found: {in_path}")

    segments_dir = Path(args.segments_dir) if args.segments_dir else (ROOT / "ir" / "segments" / in_path.stem)
    segments_dir.mkdir(parents=True, exist_ok=True)
    manifest = split_ir_by_loops(
        in_path,
        out_dir=segments_dir,
        force=bool(args.force_split),
        split_mode=str(args.split_mode),
        opt=str(args.opt),
        llvm_extract=str(args.llvm_extract),
        include_regex=str(args.include_regex),
        recursive=bool(args.recursive),
        allow_no_extract=bool(args.allow_no_extract),
    )

    print("[ok] segments dir:", str(segments_dir))
    print("[ok] manifest    :", str(manifest))


if __name__ == "__main__":
    main()
