#!/usr/bin/env python3
"""
One-shot helper: split an input LLVM IR by loops, then analyze CFG/basic blocks per segment.

This script is intentionally self-contained so other scripts can import:
  - split_ir_by_loops(...)
  - analyze_cfg_basic_blocks_for_manifest(...)
  - cfg_feature_vector_from_ll_path(...)

Example:
  python scripts/split_loops_and_analyze_cfg.py ir/dot.ll --segments-dir ir/segments/dot_demo
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


# -------------------------
# Loop splitting (outlined funcs OR loop-extract)
# -------------------------

FUNC_DEF_RE = re.compile(r"^define\s+.*@([^(]+)\(", re.IGNORECASE)
BB_LABEL_RE = re.compile(r"^\s*([A-Za-z$._][A-Za-z0-9$._-]*|\d+)\s*:\s*(?:;.*)?$")
LABEL_REF_RE = re.compile(r"label\s+%([A-Za-z$._][A-Za-z0-9$._-]*|\d+)")


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
            extracted_module.write_text(in_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

        ir_text = extracted_module.read_text(encoding="utf-8", errors="ignore")
        funcs = parse_defined_functions(ir_text)
        loop_funcs = [f for f in funcs if include_re.search(f)]
        if not loop_funcs:
            if not allow_no_extract:
                raise SystemExit(
                    "[warn] no matching segment functions found; "
                    f"mode={split_mode}, include_regex={include_regex}"
                )
            # Fallback: treat the whole (largest) function as a single "segment".
            # This is only useful when the input IR effectively has one loop, or for debugging.
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

        manifest.write_text(json.dumps(manifest_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return manifest


# -------------------------
# CFG / basic block analysis (with optional SCC)
# -------------------------

TERMINATORS = {"br", "switch", "ret", "unreachable", "resume", "indirectbr"}


def strip_comment(line: str) -> str:
    return line.split(";", 1)[0].rstrip()


def opcode_of(line: str) -> str:
    s = strip_comment(line).strip()
    if not s:
        return ""

    if "=" in s:
        left, right = s.split("=", 1)
        if left.strip().startswith("%"):
            s = right.strip()

    tokens = s.split()
    if not tokens:
        return ""

    modifiers = {
        "tail",
        "musttail",
        "notail",
        "call",
        "fastcc",
        "coldcc",
        "cc",
        "noundef",
        "nonnull",
        "noalias",
        "nocapture",
        "readonly",
        "readnone",
        "writeonly",
        "inreg",
        "byval",
        "sret",
        "align",
        "dereferenceable",
        "local_unnamed_addr",
    }
    i = 0
    while i < len(tokens) and tokens[i] in modifiers:
        i += 1
    if i >= len(tokens):
        return tokens[0]
    return tokens[i]


@dataclass
class BasicBlock:
    name: str
    instr_count: int = 0
    term: str = ""
    succs: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.succs is None:
            self.succs = []


def graph_has_cycle(nodes: list[str], succ: dict[str, list[str]]) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in nodes}
    stack: list[tuple[str, int]] = []
    for start in nodes:
        if color[start] != WHITE:
            continue
        stack.append((start, 0))
        color[start] = GRAY
        while stack:
            v, idx = stack[-1]
            outs = succ.get(v, [])
            if idx >= len(outs):
                color[v] = BLACK
                stack.pop()
                continue
            w = outs[idx]
            stack[-1] = (v, idx + 1)
            if w not in color:
                continue
            if color[w] == GRAY:
                return True
            if color[w] == WHITE:
                color[w] = GRAY
                stack.append((w, 0))
    return False


def tarjan_scc(nodes: list[str], succ: dict[str, list[str]]) -> tuple[list[list[str]], dict[str, int]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    comps: list[list[str]] = []
    node_to_comp: dict[str, int] = {}

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in succ.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            comp_id = len(comps)
            comps.append(comp)
            for n in comp:
                node_to_comp[n] = comp_id

    for n in nodes:
        if n not in indices:
            strongconnect(n)

    return comps, node_to_comp


def condensation_edges(edges: Iterable[tuple[str, str]], b2scc: dict[str, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for u, v in edges:
        su = b2scc.get(u)
        sv = b2scc.get(v)
        if su is None or sv is None or su == sv:
            continue
        out.add((su, sv))
    return out


def topo_sort_scc(num_scc: int, edges: set[tuple[int, int]]) -> list[int]:
    indeg = [0] * num_scc
    adj: list[list[int]] = [[] for _ in range(num_scc)]
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    q = [i for i in range(num_scc) if indeg[i] == 0]
    out: list[int] = []
    while q:
        u = q.pop()
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return out


def topo_sort_bb_dag(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    indeg: dict[str, int] = {n: 0 for n in nodes}
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for u, v in edges:
        if u not in adj or v not in indeg:
            continue
        adj[u].append(v)
        indeg[v] += 1
    q = [n for n in nodes if indeg[n] == 0]
    out: list[str] = []
    while q:
        u = q.pop()
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return out


def _parse_cfg_from_ll(text: str, *, scc_mode: str) -> dict[str, Any]:
    funcs: dict[str, dict[str, Any]] = {}
    cur_func: str | None = None
    cur_bb: BasicBlock | None = None

    def ensure_bb(bb_name: str) -> BasicBlock:
        nonlocal cur_bb
        f = funcs.setdefault(cur_func or "<unknown>", {"blocks": {}, "order": []})
        blocks: dict[str, BasicBlock] = f["blocks"]
        if bb_name not in blocks:
            blocks[bb_name] = BasicBlock(name=bb_name)
            f["order"].append(bb_name)
        cur_bb = blocks[bb_name]
        return cur_bb

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        m_func = FUNC_DEF_RE.match(line.strip())
        if m_func:
            cur_func = m_func.group(1)
            cur_bb = None
            funcs.setdefault(cur_func, {"blocks": {}, "order": []})
            continue

        if cur_func is None:
            continue

        if line.strip() == "}":
            cur_func = None
            cur_bb = None
            continue

        m_bb = BB_LABEL_RE.match(line)
        if m_bb:
            ensure_bb(m_bb.group(1))
            continue

        if cur_bb is None:
            continue

        s = strip_comment(line).strip()
        if not s:
            continue
        op = opcode_of(s)
        if not op:
            continue

        cur_bb.instr_count += 1

        op_l = op.lower()
        is_term = op_l in TERMINATORS or op_l == "invoke"
        if is_term:
            cur_bb.term = op_l
            succs = LABEL_REF_RE.findall(line)
            cur_bb.succs.extend(succs)

    out: dict[str, Any] = {"functions": {}}
    for fname, f in funcs.items():
        blocks: dict[str, BasicBlock] = f["blocks"]
        order: list[str] = f["order"]

        edges: list[tuple[str, str]] = []
        for b in order:
            for s in blocks[b].succs:
                edges.append((b, s))

        succ_map = {b: blocks[b].succs for b in order}
        has_cycle = graph_has_cycle(order, succ_map)

        scc_obj: dict[str, Any] | None = None
        dag_obj: dict[str, Any] | None = None
        cycle_edges: list[dict[str, str]] = []

        if scc_mode == "off":
            scc_obj = {"computed": False, "reason": "disabled", "has_cycle": has_cycle}
            if not has_cycle:
                dag_obj = {"topo_bb_order": topo_sort_bb_dag(order, edges)}
        elif scc_mode == "auto" and not has_cycle:
            scc_obj = {"computed": False, "reason": "acyclic", "has_cycle": False}
            dag_obj = {"topo_bb_order": topo_sort_bb_dag(order, edges)}
        else:
            sccs, b2scc = tarjan_scc(order, succ_map)
            cond = condensation_edges(edges, b2scc)
            topo = topo_sort_scc(len(sccs), cond)
            for u, v in edges:
                if b2scc.get(u) == b2scc.get(v):
                    cycle_edges.append({"src": u, "dst": v})
            scc_obj = {
                "computed": True,
                "has_cycle": has_cycle,
                "count": len(sccs),
                "components": [{"scc_id": i, "blocks": comp} for i, comp in enumerate(sccs)],
                "block_to_scc": {b: int(b2scc[b]) for b in order},
                "condensation_edges": [{"src_scc": u, "dst_scc": v} for (u, v) in sorted(cond)],
                "topo_scc_order": topo,
                "cycle_edges": cycle_edges,
            }

        out["functions"][fname] = {
            "basic_blocks": {
                b: {
                    "name": blocks[b].name,
                    "instr_count": blocks[b].instr_count,
                    "terminator": blocks[b].term,
                    "succs": list(blocks[b].succs),
                }
                for b in order
            },
            "order": order,
            "edges": [{"src": u, "dst": v} for u, v in edges],
            "scc": scc_obj,
            "dag": dag_obj,
            "summary": {
                "num_blocks": len(order),
                "num_edges": len(edges),
                "num_cycle_edges": len(cycle_edges),
                "has_cycle": has_cycle,
            },
        }

    return out


def _pick_main_function(funcs: dict[str, Any]) -> str | None:
    if not funcs:
        return None
    best = None
    best_bb = -1
    for name, f in funcs.items():
        order = f.get("order", []) if isinstance(f, dict) else []
        nbb = len(order) if isinstance(order, list) else 0
        if nbb > best_bb:
            best_bb = nbb
            best = name
    return best


def analyze_cfg_basic_blocks_for_manifest(
    manifest_json: str | Path,
    *,
    out_dir: str | Path,
    scc_mode: str = "auto",
) -> Path:
    """
    Analyze CFG/basic blocks for each segment in manifest.json. Writes per-segment cfg json and index.json.
    Returns index.json path.
    """
    mpath = Path(manifest_json)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    m = json.loads(mpath.read_text(encoding="utf-8"))
    segs = m.get("segments", [])

    index = {"manifest": str(mpath.resolve()), "out_dir": str(out_dir_p.resolve()), "segments": []}
    for seg in segs:
        seg_id = int(seg.get("segment_id", -1))
        fn = seg.get("func_name")
        seg_path = Path(seg["path"])
        if not seg_path.exists():
            continue
        analysis = _parse_cfg_from_ll(seg_path.read_text(encoding="utf-8", errors="ignore"), scc_mode=scc_mode)
        out_path = out_dir_p / f"{seg_id:03d}_{seg_path.stem}.cfg.json"
        out_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        index["segments"].append(
            {
                "segment_id": seg_id,
                "func_name": fn,
                "ir_path": str(seg_path.resolve()),
                "cfg_path": str(out_path.resolve()),
            }
        )

    index_path = out_dir_p / "index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return index_path


# -------------------------
# Segment-level CFG feature vector (15-d)
# -------------------------

CFG_FEAT_NAMES: list[str] = [
    "bb_count",
    "edge_count",
    "avg_succ",
    "branch_bb_frac",
    "instr_log1p",
    "has_cycle",
    "cycle_edge_frac",
    "largest_scc_frac",
]


def cfg_feature_vector_from_ll_text(text: str, *, scc_mode: str = "auto") -> tuple[np.ndarray, list[str]]:
    analysis = _parse_cfg_from_ll(text, scc_mode=scc_mode)
    funcs = analysis.get("functions", {})
    fname = _pick_main_function(funcs)
    if fname is None:
        return np.zeros((len(CFG_FEAT_NAMES),), dtype=np.float32), list(CFG_FEAT_NAMES)

    f = funcs[fname]
    order = f.get("order", [])
    edges = f.get("edges", [])
    bbs = f.get("basic_blocks", {})

    bb_count = float(len(order)) if isinstance(order, list) else 0.0
    edge_count = float(len(edges)) if isinstance(edges, list) else 0.0
    avg_succ = (edge_count / bb_count) if bb_count > 0 else 0.0

    branch_bbs = 0
    total_instr = 0.0

    if isinstance(order, list) and isinstance(bbs, dict):
        for b in order:
            bb = bbs.get(b, {})
            succs = bb.get("succs", [])
            if isinstance(succs, list) and len(succs) > 1:
                branch_bbs += 1
            ic = bb.get("instr_count", 0)
            try:
                total_instr += float(ic)
            except Exception:
                pass

    branch_bb_frac = (float(branch_bbs) / bb_count) if bb_count > 0 else 0.0
    instr_log1p = math.log1p(total_instr) if total_instr > 0 else 0.0

    scc = f.get("scc", {}) if isinstance(f, dict) else {}
    has_cycle = bool(scc.get("has_cycle", f.get("summary", {}).get("has_cycle", False)))
    num_cycle_edges = 0.0
    largest_scc = 1.0
    if isinstance(scc, dict) and bool(scc.get("computed", False)):
        cycle_edges = scc.get("cycle_edges", [])
        if isinstance(cycle_edges, list):
            num_cycle_edges = float(len(cycle_edges))
        comps = scc.get("components", [])
        if isinstance(comps, list) and comps:
            largest_scc = float(
                max((len(c.get("blocks", [])) for c in comps if isinstance(c, dict)), default=1)
            )
    else:
        largest_scc = 1.0

    cycle_edge_frac = (num_cycle_edges / edge_count) if edge_count > 0 else 0.0
    largest_scc_frac = (largest_scc / bb_count) if bb_count > 0 else 0.0

    vec = np.asarray(
        [
            bb_count,
            edge_count,
            avg_succ,
            branch_bb_frac,
            instr_log1p,
            1.0 if has_cycle else 0.0,
            cycle_edge_frac,
            largest_scc_frac,
        ],
        dtype=np.float32,
    )
    return vec, list(CFG_FEAT_NAMES)


def cfg_feature_vector_from_ll_path(path: str | Path, *, scc_mode: str = "auto") -> tuple[np.ndarray, list[str]]:
    p = Path(path)
    return cfg_feature_vector_from_ll_text(p.read_text(encoding="utf-8", errors="ignore"), scc_mode=scc_mode)


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
    p.add_argument(
        "--cfg-out",
        default=None,
        help="CFG analysis output dir (default: <segments-dir>/cfg)",
    )
    p.add_argument(
        "--no-cfg",
        action="store_true",
        help="only split loops; do not write cfg/*.cfg.json outputs",
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
        "--scc-mode",
        default="auto",
        choices=["auto", "always", "off"],
        help="SCC computation mode for CFG analysis",
    )

    p.add_argument(
        "--save-cfg-features-csv",
        default=None,
        help="optional path to save per-segment CFG summary features as CSV",
    )
    p.add_argument(
        "--save-cfg-features-npz",
        default=None,
        help="optional path to save per-segment CFG summary features as NPZ",
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
    )

    cfg_out = Path(args.cfg_out) if args.cfg_out else (segments_dir / "cfg")
    if not args.no_cfg:
        cfg_out.mkdir(parents=True, exist_ok=True)
        analyze_cfg_basic_blocks_for_manifest(manifest, out_dir=cfg_out, scc_mode=str(args.scc_mode))

    # Optional: emit the model's 15-d CFG summary features per segment.
    save_csv = args.save_cfg_features_csv is not None
    save_npz = args.save_cfg_features_npz is not None
    if save_csv or save_npz:
        m = json.loads(manifest.read_text(encoding="utf-8"))
        segs = m.get("segments", [])

        rows = []
        X = []
        for seg in segs:
            seg_id = int(seg.get("segment_id", -1))
            fn = str(seg.get("func_name", ""))
            seg_path = Path(seg["path"])
            v, _ = cfg_feature_vector_from_ll_path(seg_path, scc_mode="auto")
            rows.append((seg_id, fn, str(seg_path)))
            X.append(v)

        X_np = np.stack(X, axis=0) if X else np.zeros((0, len(CFG_FEAT_NAMES)), dtype=np.float32)

        if save_csv:
            out_csv = Path(args.save_cfg_features_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["segment_id", "func_name", "ir_path", *CFG_FEAT_NAMES])
                for (seg_id, fn, path), vec in zip(rows, X_np):
                    w.writerow([seg_id, fn, path, *[float(x) for x in vec.tolist()]])

        if save_npz:
            out_npz = Path(args.save_cfg_features_npz)
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_npz,
                X=X_np,
                ids=np.array([f"{seg_id}:{fn}" for (seg_id, fn, _) in rows]),
                feat_names=np.array(CFG_FEAT_NAMES),
            )

    print("[ok] segments dir:", str(segments_dir))
    print("[ok] manifest    :", str(manifest))
    if not args.no_cfg:
        print("[ok] cfg out     :", str(cfg_out))


if __name__ == "__main__":
    main()
