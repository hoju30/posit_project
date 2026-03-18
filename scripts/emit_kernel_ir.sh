#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IR_DIR="$ROOT/ir"
SRC_DIR="$ROOT/src/ir_kernels"

mkdir -p "$IR_DIR"

CLANG="${CLANG:-clang-16}"
CXXFLAGS=(
  -std=c++20
  -O2
  -S
  -emit-llvm
  -Xclang
  -no-opaque-pointers
)

emit() {
  local name="$1"
  local src="$SRC_DIR/$name.cpp"
  local out="$IR_DIR/$name.ll"
  echo "[emit] $out"
  "$CLANG" "${CXXFLAGS[@]}" "$src" -o "$out"
}

emit dot
emit sum
emit axpy_sum
emit relu_sum
emit l1_norm
emit sum_squares
emit axpby_sum
emit matvec
emit prefix_sum

