import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import joblib

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

def main():
    data = np.load(DATA_DIR / "ml_dataset_ir_errors.npz", allow_pickle=True)
    X = data["X"]
    Y = data["Y"]
    target_names = data["target_names"].tolist()
    meta = data.get("meta", None)

    # 保留目標為：posit_* 與 fp32（皆為相對 fp64 的 rel_err）

    eps = 1e-30
    Y_log = np.log10(np.maximum(Y, 0.0) + eps)

    reg = RandomForestRegressor(
        n_estimators=800,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=0,
        n_jobs=-1,
    )
    reg.fit(X, Y_log)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": reg,
            "target_names": target_names,
            "meta": dict(meta.item()) if meta is not None else None,
            "y_transform": {"type": "log10", "eps": eps},
        },
        MODEL_DIR / "ir2vec_error_predictor.joblib"
    )

if __name__ == "__main__":
    main()
