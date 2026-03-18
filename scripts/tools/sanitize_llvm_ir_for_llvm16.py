#!/usr/bin/env python3
"""
Sanitize LLVM IR text to be parseable by LLVM 16 tools (opt-16 / llvm-extract-16 / ir2vec).

Motivation
----------
If you generate LLVM IR with a newer LLVM (e.g. via mlir-translate from LLVM 19/21),
the IR may contain syntax that LLVM 16 rejects. One common example is:

  getelementptr inbounds nuw float, ptr %p, i64 %i

LLVM 16 does not accept the `nuw`/`nsw` flags on GEP, so downstream tooling fails.

This script applies a conservative rewrite to remove those unsupported flags.
It is intended for *analysis / feature extraction* workflows, not for bit-exact codegen.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def sanitize(text: str) -> str:
    # Drop nuw/nsw on getelementptr: "getelementptr inbounds nuw nsw T, ..."
    # -> "getelementptr inbounds T, ..."
    gep_pat = re.compile(r"(getelementptr\s+inbounds)(?:\s+(?:nuw|nsw))+(\s+)", re.IGNORECASE)
    text = gep_pat.sub(r"\1\2", text)
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("in_ll", help="input LLVM IR (.ll)")
    p.add_argument("-o", "--out", required=True, help="output sanitized .ll")
    args = p.parse_args()

    in_path = Path(args.in_ll)
    out_path = Path(args.out)
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")

    raw = in_path.read_text(encoding="utf-8", errors="ignore")
    fixed = sanitize(raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(fixed, encoding="utf-8")
    print("[ok] wrote:", str(out_path))


if __name__ == "__main__":
    main()

