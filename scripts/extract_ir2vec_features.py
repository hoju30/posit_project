# extract_ir2vec_features.py
import os
import numpy as np
import ir2vec
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IR_DIR = ROOT / "ir"
DATA_DIR = ROOT / "data"

# 依 kernel.cpp 中的運算列出預期的 kernel 名稱
KERNELS = [
    "dot",
    "sum",
    "axpy_sum",
    "relu_sum",
    "l1_norm",
    "sum_squares",
    "axpby_sum",
    "matvec",
    "prefix_sum",
]


def find_ir_path(kernel_id: str) -> str:

    candidates = [
        IR_DIR / f"{kernel_id}.ll",
        IR_DIR / f"{kernel_id}.bc",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"IR for kernel '{kernel_id}' not found under {IR_DIR} (.ll or .bc)")


def main():
    xs = []
    ids = []

    for kernel_id in KERNELS:
        path = find_ir_path(kernel_id)
        init_obj = ir2vec.initEmbedding(path, "fa", "p")

        prog_vec = init_obj.getProgramVector()  # ndarray, shape = (dim,)
        prog_vec = np.asarray(prog_vec, dtype=np.float32)

        xs.append(prog_vec)
        ids.append(kernel_id)

    X = np.stack(xs, axis=0)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_DIR / "ir2vec_features.npz", X=X, ids=np.array(ids))


if __name__ == "__main__":
    main()
