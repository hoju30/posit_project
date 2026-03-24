# scripts

這個目錄分成兩條主線：

1. `regression model` 主線  
用 kernel dataset + IR2Vec 建立回歸資料、訓練 per-format error predictor、做 in-domain 評估與 ablation。

2. `loop / MLIR` 工具主線  
把 LLVM IR 依 loop 切段、產生 per-loop prediction、再轉成 MLIR plan 給 pass 使用。

注意：**目前正式回歸模型已經不吃 CFG features**。  
`split_loops_and_analyze_cfg.py` 仍保留，因為它同時負責 loop segmentation；但現在 regression dataset 與單檔推論路徑不再把 CFG 向量接進 model。

## 回歸模型主線

建議流程：

1. `emit_kernel_ir.sh`  
用 `clang-16` 產生乾淨的 kernel LLVM IR，輸出到 `ir/*.ll`。

2. `extract_ir2vec_features.py`  
把 `ir/*.ll` 轉成 `data/ir2vec_features.npz`。

3. `build_ml_dataset_ir_errors.py`  
把 `data/*_dataset.csv` 和 `ir2vec_features.npz` 組成 `data/ml_dataset_ir_errors.npz`。  
目前輸出的是 **per-format regression dataset**：
   - `Y` 是單一 scalar relative error
   - `X` 包含 `IR2Vec + scalar + raw/derived stats + format features`
   - 不含 CFG

4. `train_ir_error_model.py`  
訓練 `models/ir2vec_error_predictor.joblib`。  
目前是：
   - 每個 posit format 各自一顆 model
   - 內建 `selective calibration`

5. `predict_ir_errors.py`  
對單一 IR 預測全部 posit format 的誤差。  
會自動讀取 model bundle 內的 selective calibration 設定。

## 評估與繪圖

- `eval_in_domain_old_vs_current.py`  
公平比較舊架構 baseline 與目前模型。  
可比較：
  - `old_ir2vec_stats_multioutput`
  - `current_per_format_models_quant`
  - calibrated / selective calibrated 版本

- `eval_feature_ablation_current.py`  
對目前模型做 feature ablation。  
目前主要比較：
  - `ir2vec_only`
  - `ir2vec_stats`
  - `full_no_quant`
  - `full_quant`


## Per-loop / MLIR 主線

- `split_loops.py`  
把 LLVM IR 依 loop 切段，並輸出 segment manifest，用途是 **loop segmentation**。

- `predict_ir_errors_by_loops.py`  
對每個 loop segment 預測各 posit format 誤差，輸出 `predictions.json`。  
目前也會自動套 model bundle 內的 selective calibration。

- `run_loop_pipeline.py`  
單一入口，串起：
  - split loop segments
  - per-loop prediction
  - output `predictions.json`

## 其他共用腳本

- `error_feature_utils.py`  
共用的 feature helper。  
包含：
  - raw stats 欄位定義
  - derived posit-aware stats
  - quantization-aware feature 名稱
  - format name parsing

## tools/

`scripts/tools/` 放的是支援 loop / MLIR pass 的工具腳本：

- `sanitize_llvm_ir_for_llvm16.py`：把 LLVM IR 清成較穩定可處理的版本
- `gen_mlir_loop_plan.py`：從 MLIR loop 產生外部 plan
- `build_loop_plan_from_marked_segments.py`：把 segment prediction 對回 `loop_id`，產出 `plan.json`
- `build_loop_plan_from_segments.py`：較舊的 segment → plan 腳本

詳細可再看：
- [tools/README.md](/home/hoju/test/posit_test/scripts/tools/README.md)
