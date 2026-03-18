# scripts

常用入口（回歸 / per-loop / CFG）：

- `run_loop_pipeline.py`：單一入口（MLIR → LLVM loop segments → per-loop 預測 → plan.json → 回貼 MLIR）
- `emit_kernel_ir.sh`：用 clang-16 產生乾淨的 kernel LLVM IR（`ir/*.ll`）
- `extract_ir2vec_features.py`：把 `ir/*.ll` 轉成 `data/ir2vec_features.npz`
- `build_ml_dataset_ir_errors.py`：把 `data/*_dataset.csv` 組成回歸訓練資料 `data/ml_dataset_ir_errors.npz`
- `train_ir_error_model.py`：訓練回歸模型 `models/ir2vec_error_predictor.joblib`
- `predict_ir_errors.py`：對單一 IR 預測各 datatype 的相對誤差（輸出可存 CSV/JSON）
- `split_loops_and_analyze_cfg.py`：把 LLVM IR 依 loop 切段 + CFG/SCC 分析 + 8 維 CFG 彙總特徵
- `predict_ir_errors_by_loops.py`：逐段 loop segment 推論（輸出 `predictions.json`）
- `plot_bar.py`：把 `target,pred` CSV 畫成 bar 圖

工具腳本放在 `scripts/tools/`（MLIR plan、LLVM IR sanitize、segment↔loop_id 對應等）。
