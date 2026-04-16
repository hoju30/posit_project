# Regression Pipeline（誤差預測）

此分支專注於從 LLVM IR 預測各 posit 格式（8/16/32, es≤2）對 fp64 的相對誤差。

目前模型是 `per-format regression`：
- 每個 posit format 各自一顆 model。
- 推論時對同一個輸入依序跑全部 format，再輸出整張誤差表。
- 主線有兩種模型：
  - `ir2vec_error_predictor.joblib`：RandomForest baseline
  - `ir2vec_error_predictor_xgb.joblib`：XGBoost 主線
- 兩者目前都使用 `log-target + selective calibration`。

目前回歸模型的輸入特徵（X）：
- `IR2Vec` 程式向量（300 維）
- `shared features`
  - scalar：`vec_len/rows/cols/alpha/beta`
  - raw stats：`mean/std/min/max/abs_max/skewness/excess_kurtosis/p01/p50/p99/near_zero_ratio/pos_ratio/neg_ratio`
  - derived stats：`log10_std/upper_tail/lower_tail/standardized_mean/center_decade/span_decades`
- `format-aware features`
  - `current-es excess`：`upper/lower_excess_current_es`
  - quant features：`oor_ratio/clip_high_ratio/clip_low_ratio/rel_qerr_mean/zero_after_quant_ratio/unique_ratio`

另外，專案目前也開始準備 `variable-level mixed-precision` 路線：
- whole-app LLVM IR entity 抽取腳本：[scripts/extract_entity_features.py](/home/hoju/test/posit_test/scripts/extract_entity_features.py)
- generated-app actual label 回填腳本：[scripts/fill_entity_labels.py](/home/hoju/test/posit_test/scripts/fill_entity_labels.py)
- 目前先抓：
  - `accumulator`
  - `loop_carried_state`
  - `reduction_intermediate`
- 目前先抽：
  - entity-local：`role / def_opcode / use_count / is_loop_carried / is_output_related`
  - graph-derived：`fanout / reduction_depth / distance_to_output / same_loop_neighbor_count / num_arith_users`
- 目前 first cut 的 actual label 只先做 `accumulator-only`：
  - 其他 entity 固定 `fp64`
  - target 是 whole-app `actual_whole_app_rel_err`
  - binary label 是 `is_feasible_under_tol = (actual_whole_app_rel_err <= 1e-3)`
  - 目前全量 generated-app corpus 中，這條路已可直接回填 `150 apps`

目前正式特徵維度是：
- `359 = 300 IR2Vec + 43 shared + 16 format-aware`

實作位置：
- raw stats / quant features：[src/stats.hpp](/home/hoju/test/posit_test/src/stats.hpp)
- derived / current-es features：[scripts/error_feature_utils.py](/home/hoju/test/posit_test/scripts/error_feature_utils.py)
- dataset 組裝：[scripts/build_ml_dataset_ir_errors.py](/home/hoju/test/posit_test/scripts/build_ml_dataset_ir_errors.py)

XGBoost 目前採用的較佳參數：
- `n_estimators=600`
- `max_depth=4`
- `learning_rate=0.02`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `reg_lambda=4.0`
- `min_child_weight=5.0`

## 步驟
1. **C++ 產生資料**（共用 `bin/*.run`，輸出 `data/*_dataset.csv`）。
   - dot：`g++ -std=c++20 -O2 src/main_dot.cpp src/kernel.cpp -Iexternal/universal/include/sw -o bin/dot_run`
   - 產生：`./bin/dot_run --random-scale`（預設開啟；可用 `--fixed-scale` 回到原本固定尺度）
2. **產生 LLVM IR**（clang-16 `-no-opaque-pointers`，IR 放 `ir/*.ll`）。
   - 建議用純 kernel IR：`bash scripts/emit_kernel_ir.sh`
   - 若你的輸入是一個「大型 IR」且包含多個 loop：可先用 opt 把 loop 抽成多段 IR，再逐段做 ir2vec  
     `python scripts/split_loops_and_analyze_cfg.py path/to/input.ll --segments-dir ir/segments/input --no-cfg`
     - 會輸出 `manifest.json`（每個 loop segment 的 `.ll` 路徑清單）
3. **抽取 ir2vec 特徵**  
   `python scripts/extract_ir2vec_features.py` → `data/ir2vec_features.npz`
4. **組合誤差資料集**（以 sample_id 展開，每設定最多 200 筆）  
   `python scripts/build_ml_dataset_ir_errors.py` → `data/ml_dataset_ir_errors.npz`
5. **訓練回歸模型**  
   `python scripts/train_ir_error_model.py` → `models/ir2vec_error_predictor.joblib`
   `python scripts/train_ir_error_model_xgb.py` → `models/ir2vec_error_predictor_xgb.joblib`
6. **推論誤差**  
   `python scripts/predict_ir_errors.py ir/dot.ll --vec-len 128 --x-mean 0 --x-std 1 --x-p1 -3 --x-p50 0 --x-p99 3 --y-mean 0 --y-std 10 --y-p1 -30 --y-p50 0 --y-p99 30 --save-csv data/dot_error_preds.csv`
7. **繪圖（可選）**  
   `python scripts/plot_bar.py --csv data/dot_error_preds.csv --out-prefix data/dot_error_bar`

## 評估
- RandomForest vs XGBoost：
  `python scripts/eval_in_domain_xgb.py`
- XGBoost 調參：
  `python scripts/tune_xgb_in_domain.py`
- benchmark 對照：
  - `python scripts/eval_atax_benchmark.py`
  - `python scripts/eval_bicg_benchmark.py`
  會輸出 `actual.json / pred.json / compare.json / compare.csv`，可檢查 model prediction 和 benchmark 真值的差距。

## 混合精度（per-loop）推論
當你輸入的是「大型 IR」且包含多個 loop，建議用 per-loop 的方式避免全域 IR2Vec 向量混到不相關的訊號：

1. **依 loop 切段**（會輸出 `manifest.json` + 多個 `.ll` segment）  
   `python scripts/split_loops_and_analyze_cfg.py path/to/input.ll --segments-dir ir/segments/input`
2. **逐段推論 + 產生混合精度決策（不改 IR，只輸出 plan）**  
   `python scripts/predict_ir_errors_by_loops.py path/to/input.ll --vec-len 128 --x-mean 0 --x-std 1 --x-p1 -3 --x-p50 0 --x-p99 3 --y-mean 0 --y-std 10 --y-p1 -30 --y-p50 0 --y-p99 30 --pick min_bits_under_tol --tol 1e-3`

輸出：
- `ir/segments/<stem>/predictions.json`：每個 loop segment 的誤差預測 + `chosen`（該段建議格式）
- 可加 `--save-csv data/<stem>_agg.csv` 產生聚合後的 `target,pred`，再用 `plot_bar.py` 畫圖

## 輸出
- 模型：`models/ir2vec_error_predictor.joblib`
- XGBoost 模型：`models/ir2vec_error_predictor_xgb.joblib`
- 推論腳本：`scripts/predict_ir_errors.py`
- 其他：誤差長條圖 `dot_error_bar*.png`、預測 CSV。

> 資料生成與 IR 產生與分類管線共用，流程細節可參考專案根的 README。
