#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FUNC_DEF_RE = re.compile(r"^define\s+.*@([^(]+)\(")
BB_RE = re.compile(r"^([A-Za-z$._][-A-Za-z$._0-9]*):")
INST_RE = re.compile(r"^\s*(%[-A-Za-z$._0-9]+)\s*=\s*([A-Za-z][A-Za-z0-9._-]*)\b(.*)$")
SSA_RE = re.compile(r"(%[-A-Za-z$._0-9]+)")

ARITH_OPS = {
    "fadd", "fsub", "fmul", "fdiv", "frem",
    "add", "sub", "mul", "sdiv", "udiv", "srem", "urem",
    "shl", "lshr", "ashr", "and", "or", "xor",
    "icmp", "fcmp", "select", "fneg",
}
SINK_OPS = {"ret", "store"}


@dataclass
class Instruction:
    inst_id: str
    opcode: str
    operands: list[str]
    bb: str
    raw: str
    name: str | None = None
    users: set[str] = field(default_factory=set)
    operand_defs: set[str] = field(default_factory=set)


def safe_float(v: int | float) -> float | int:
    return int(v) if isinstance(v, int) else float(v)


def normalized_opcode(raw: str, parsed_opcode: str) -> str:
    text = raw.strip()
    if "@llvm.fmuladd." in text:
        return "fmuladd"
    if "@llvm.fma." in text:
        return "fma"
    if "@llvm.fabs." in text:
        return "fabs"
    if parsed_opcode in {"tail", "call", "musttail", "notail"}:
        m = re.search(r"@([A-Za-z0-9$._]+)", text)
        if m:
            callee = m.group(1)
            if callee.startswith("llvm.fmuladd"):
                return "fmuladd"
            if callee.startswith("llvm.fma"):
                return "fma"
            if callee.startswith("llvm.fabs"):
                return "fabs"
            return callee
    return parsed_opcode


def is_fp_like_raw(raw: str) -> bool:
    markers = (" float ", " double ", "<2 x double>", "<2 x float>", "<4 x double>", "<4 x float>", "<8 x double>", "<8 x float>")
    return any(m in f" {raw} " for m in markers)


def is_fp_arith_opcode(opcode: str) -> bool:
    return opcode.startswith("f") or opcode in {"fmuladd", "fma"}


def parse_functions(ir_text: str) -> list[dict[str, Any]]:
    lines = ir_text.splitlines()
    funcs: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        m = FUNC_DEF_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        func_name = m.group(1)
        body: list[str] = [lines[i]]
        depth = lines[i].count("{") - lines[i].count("}")
        i += 1
        while i < len(lines):
            body.append(lines[i])
            depth += lines[i].count("{") - lines[i].count("}")
            if depth <= 0:
                i += 1
                break
            i += 1
        funcs.append({"func_name": func_name, "lines": body})
    return funcs


def build_instruction_graph(func: dict[str, Any]) -> tuple[list[Instruction], dict[str, str]]:
    instructions: list[Instruction] = []
    def_to_inst: dict[str, str] = {}
    current_bb = "entry"
    local_idx = 0

    for line in func["lines"]:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        bb_m = BB_RE.match(stripped)
        if bb_m:
            current_bb = bb_m.group(1)
            continue
        if stripped.startswith("define") or stripped == "}":
            continue

        inst_id = f"{func['func_name']}::i{local_idx}"
        local_idx += 1
        m = INST_RE.match(line)
        if m:
            name, opcode, rest = m.groups()
            opcode = normalized_opcode(stripped, opcode)
            operands = SSA_RE.findall(rest)
            inst = Instruction(
                inst_id=inst_id,
                name=name,
                opcode=opcode,
                operands=operands,
                bb=current_bb,
                raw=stripped,
            )
            instructions.append(inst)
            def_to_inst[name] = inst_id
            continue

        opcode = normalized_opcode(stripped, stripped.split(None, 1)[0])
        operands = SSA_RE.findall(stripped)
        instructions.append(
            Instruction(
                inst_id=inst_id,
                name=None,
                opcode=opcode,
                operands=operands,
                bb=current_bb,
                raw=stripped,
            )
        )

    inst_by_id = {inst.inst_id: inst for inst in instructions}
    for inst in instructions:
        for operand in inst.operands:
            src_id = def_to_inst.get(operand)
            if src_id is None:
                continue
            inst.operand_defs.add(src_id)
            inst_by_id[src_id].users.add(inst.inst_id)

    return instructions, def_to_inst


def sink_ids(instructions: list[Instruction]) -> set[str]:
    sinks = {inst.inst_id for inst in instructions if inst.opcode in SINK_OPS}
    if sinks:
        return sinks
    return {inst.inst_id for inst in instructions if not inst.users}


def shortest_distance_to_sink(start_id: str, inst_by_id: dict[str, Instruction], sinks: set[str]) -> int:
    if start_id in sinks:
        return 0
    queue: list[tuple[str, int]] = [(start_id, 0)]
    seen = {start_id}
    head = 0
    while head < len(queue):
        node, dist = queue[head]
        head += 1
        for user in inst_by_id[node].users:
            if user in seen:
                continue
            if user in sinks:
                return dist + 1
            seen.add(user)
            queue.append((user, dist + 1))
    return -1


def output_related_set(inst_by_id: dict[str, Instruction], sinks: set[str]) -> set[str]:
    related = set(sinks)
    queue = list(sinks)
    while queue:
        node = queue.pop()
        for pred in inst_by_id[node].operand_defs:
            if pred in related:
                continue
            related.add(pred)
            queue.append(pred)
    return related


def arithmetic_depth(start_id: str, inst_by_id: dict[str, Instruction], memo: dict[str, int], stack: set[str]) -> int:
    if start_id in memo:
        return memo[start_id]
    if start_id in stack:
        return 0
    stack.add(start_id)
    best = 0
    for user in inst_by_id[start_id].users:
        user_inst = inst_by_id[user]
        if user_inst.opcode not in ARITH_OPS:
            continue
        best = max(best, 1 + arithmetic_depth(user, inst_by_id, memo, stack))
    stack.remove(start_id)
    memo[start_id] = best
    return best


def detect_accumulators(instructions: list[Instruction], inst_by_id: dict[str, Instruction]) -> set[str]:
    accs: set[str] = set()
    for inst in instructions:
        if inst.opcode != "phi" or not inst.name:
            continue
        if not is_fp_like_raw(inst.raw):
            continue
        for user_id in inst.users:
            user = inst_by_id[user_id]
            if not user.name or user.opcode not in {"fadd", "fmuladd", "fma"}:
                continue
            if inst.name in user.operands and user.name in inst.operands:
                accs.add(inst.inst_id)
                break
    return accs


def classify_entities(instructions: list[Instruction]) -> list[dict[str, Any]]:
    inst_by_id = {inst.inst_id: inst for inst in instructions}
    sinks = sink_ids(instructions)
    out_related = output_related_set(inst_by_id, sinks)
    accs = detect_accumulators(instructions, inst_by_id)
    depth_memo: dict[str, int] = {}

    rows: list[dict[str, Any]] = []
    for inst in instructions:
        if not inst.name:
            continue

        role: str | None = None
        is_loop_carried = inst.opcode == "phi"
        if inst.inst_id in accs:
            role = "accumulator"
        elif is_loop_carried:
            role = "loop_carried_state"
        elif is_fp_arith_opcode(inst.opcode) and inst.inst_id in out_related:
            role = "reduction_intermediate"

        if role is None:
            continue

        neighbor_ids = set(inst.operand_defs) | set(inst.users)
        num_arith_users = sum(1 for uid in inst.users if inst_by_id[uid].opcode in ARITH_OPS)
        rows.append(
            {
                "entity": inst.name,
                "role": role,
                "def_opcode": inst.opcode,
                "use_count": safe_float(len(inst.users)),
                "is_loop_carried": safe_float(1 if is_loop_carried else 0),
                "is_output_related": safe_float(1 if inst.inst_id in out_related else 0),
                "fanout": safe_float(len(inst.users)),
                "reduction_depth": safe_float(arithmetic_depth(inst.inst_id, inst_by_id, depth_memo, set())),
                "distance_to_output": safe_float(shortest_distance_to_sink(inst.inst_id, inst_by_id, sinks)),
                "same_loop_neighbor_count": safe_float(len(neighbor_ids)),
                "num_arith_users": safe_float(num_arith_users),
                "bb": inst.bb,
                "raw": inst.raw,
            }
        )
    return rows


def analyze_ir(ir_path: Path) -> dict[str, Any]:
    ir_text = ir_path.read_text(encoding="utf-8", errors="ignore")
    functions = parse_functions(ir_text)
    out_funcs: list[dict[str, Any]] = []
    for func in functions:
        instructions, _ = build_instruction_graph(func)
        entities = classify_entities(instructions)
        if not entities:
            continue
        out_funcs.append(
            {
                "func_name": func["func_name"],
                "entity_count": len(entities),
                "entities": entities,
            }
        )
    return {
        "input_ir": str(ir_path.resolve()),
        "scope": "whole_app_module",
        "same_loop_note": "same_loop_neighbor_count currently approximates same-function neighbors until explicit loop analysis is added",
        "function_count": len(out_funcs),
        "functions": out_funcs,
    }


def flatten_rows(obj: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fn in obj.get("functions", []):
        for entity in fn["entities"]:
            rows.append(
                {
                    "function": fn["func_name"],
                    **entity,
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "function",
        "entity",
        "role",
        "def_opcode",
        "use_count",
        "is_loop_carried",
        "is_output_related",
        "fanout",
        "reduction_depth",
        "distance_to_output",
        "same_loop_neighbor_count",
        "num_arith_users",
        "bb",
        "raw",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ir", help="input LLVM IR (.ll)")
    ap.add_argument("--out-json", default=None, help="write analysis JSON")
    ap.add_argument("--out-csv", default=None, help="write flattened CSV")
    args = ap.parse_args()

    in_path = Path(args.ir)
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")

    result = analyze_ir(in_path)
    out_json = Path(args.out_json) if args.out_json else in_path.with_suffix(".entities.json")
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {out_json}")

    if args.out_csv:
        rows = flatten_rows(result)
        out_csv = Path(args.out_csv)
        write_csv(rows, out_csv)
        print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
