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

5. `train_ir_error_model_xgb.py`
   訓練 `models/ir2vec_error_predictor_xgb.joblib`。  
   目前主線採用的較佳參數是：
   - `n_estimators=600`
   - `max_depth=4`
   - `learning_rate=0.02`
   - `subsample=0.8`
   - `colsample_bytree=0.8`
   - `reg_lambda=4.0`
   - `min_child_weight=5.0`

6. `predict_ir_errors.py`  
   對單一 IR 預測全部 posit format 的誤差。  
   會自動讀取 model bundle 內的 selective calibration 設定。

## Variable-level / whole-app 草稿路線

- `generate_source_apps.py`
  產生 whole-app source corpus。  
  目前預設：
  - `8 families`
  - `25 variants / family`
  - 共 `200 apps`

  families 目前是：
  - `sum_reduce`
  - `dot_product`
  - `weighted_sum`
  - `prefix_recurrence`
  - `ema_recurrence`
  - `matvec_like`
  - `two_stage_reduce`
  - `stencil1d_like`

  產物會放在：
  - `generated_apps/src/<family>/*.c`
  - `generated_apps/apps_manifest.csv`
  - `generated_apps/summary.json`

- `extract_entity_features.py`
  從整個 app 的 LLVM IR 抽 variable-level entities，先聚焦三類：
  - `accumulator`
  - `loop_carried_state`
  - `reduction_intermediate`

  目前輸出的 entity-local / graph-derived features 包含：
  - `role`
  - `def_opcode`
  - `use_count`
  - `is_loop_carried`
  - `is_output_related`
  - `fanout`
  - `reduction_depth`
  - `distance_to_output`
  - `same_loop_neighbor_count`
  - `num_arith_users`

  注意：
  - 現在是 whole-app text-level LLVM IR 分析
  - 還沒有真的接 pass 做 variable-level type rewrite
  - `same_loop_neighbor_count` 目前先用「same-function def-use neighbors」近似，之後再接真正 loop analysis

- `generate_entity_labels.py`
  產生 entity-level binary classification dataset 的骨架。  
  目前先支援 `accumulator` 類別，展開成：
  - `(app, function, entity, format) -> feasible?`
  - threshold 先固定看 `l2_rel_err <= 1e-3`
  - 其他 entity 先假設維持 `fp64`

  目前這支只會先把 schema 和 candidate rows 建好，輸出欄位包含：
  - entity features
  - `format / format_n / format_es`
  - `entity_id`
  - `actual_whole_app_rel_err`
  - `is_feasible_under_tol`
  - `margin_to_tol`
  - `label_status`

  其中：
  - `actual_whole_app_rel_err`
  - `is_feasible_under_tol`
  - `margin_to_tol`
  初始 skeleton 會先留空，之後再由 actual evaluator 回填。

- `compile_generated_apps.py`
  把 generated apps 編成 LLVM IR，並串起：
  - `extract_entity_features.py`
  - `generate_entity_labels.py`

  會輸出：
  - `generated_apps/compiled/ir/...`
  - `generated_apps/compiled/entity_features/...`
  - `generated_apps/compiled/entity_labels/...`
  - `generated_apps/compiled/compiled_manifest.csv`

- `fill_entity_labels.py`
  對 generated apps 的 entity label skeleton 回填 actual label。  
  目前先支援：
  - `accumulator`
  - `loop_carried_state`
  - `reduction_intermediate`
  - 其他 entity 固定 `fp64`
  - `actual_whole_app_rel_err`
  - `is_feasible_under_tol = (actual_whole_app_rel_err <= 1e-3)`
  - `margin_to_tol = 1e-3 - actual_whole_app_rel_err`

  注意：
  目前 first cut coverage：
  - `accumulator`
    - `150 apps`
    - families:
      - `sum_reduce`
      - `dot_product`
      - `weighted_sum`
      - `prefix_recurrence`
      - `two_stage_reduce`
      - `stencil1d_like`
    - `ema_recurrence` 目前在 O1 IR 下沒有 accumulator rows；`matvec_like` 目前是 `2 accumulator rows`，先保留 skeleton、不自動回填
  - `loop_carried_state`
    - `25 apps`
    - family:
      - `ema_recurrence`
  - `reduction_intermediate`
    - `150 apps`
    - families:
      - `dot_product`
      - `weighted_sum`
      - `prefix_recurrence`
      - `ema_recurrence`
      - `two_stage_reduce`
      - `stencil1d_like`
  - source-level evaluator 目前有：
    - [generated_app_accumulator_eval.cpp](/home/hoju/test/posit_test/src/generated_app_accumulator_eval.cpp)
    - [generated_app_reduction_intermediate_eval.cpp](/home/hoju/test/posit_test/src/generated_app_reduction_intermediate_eval.cpp)

- `merge_entity_label_datasets.py`
  把多個已回填的 entity label CSV 合併成單一 train CSV。  
  目前可直接把 `generated_apps/compiled_accum_O1` 合成：
  - `data/generated_entity_label_dataset_accum.csv`

- `merge_role_entity_datasets.py`
  把三類 entity 的 train CSV 再合成 unified dataset。  
  目前預設會合併：
  - `data/generated_entity_label_dataset_accum.csv`
  - `data/generated_entity_label_dataset_lcs.csv`
  - `data/generated_entity_label_dataset_ri.csv`
  產生：
  - `data/generated_entity_label_dataset_all.csv`
  目前 merged rows：
  - `2925`

- `train_entity_feasibility_model.py`
  訓練第一版 entity-level binary classifier。  
  可直接拿：
  - `data/generated_entity_label_dataset_accum.csv`
  - `data/generated_entity_label_dataset_lcs.csv`
  - `data/generated_entity_label_dataset_ri.csv`
  - `data/generated_entity_label_dataset_all.csv`
  任何一份 dataset 來訓練，並用 group-by-app holdout split。
  目前 unified RF holdout：
  - `accuracy = 0.945299`
  - `f1 = 0.952941`
  - `roc_auc = 0.991090`

- `train_entity_feasibility_model_xgb.py`
  訓練 XGBoost 版本的 entity-level binary classifier。  
  目前同樣可直接拿：
  - `data/generated_entity_label_dataset_accum.csv`
  - `data/generated_entity_label_dataset_lcs.csv`
  - `data/generated_entity_label_dataset_ri.csv`
  - `data/generated_entity_label_dataset_all.csv`
  任何一份 dataset 來訓練，並用 group-by-app holdout split。
  目前 unified XGB holdout：
  - `accuracy = 0.945299`
  - `f1 = 0.952941`
  - `roc_auc = 0.990035`

- `eval_entity_feasibility_unified_features.py`
  對 unified entity dataset 做 feature importance 與 feature-group ablation。  
  目前輸出在：
  - `data/eval_entity_feasibility_unified_features/ablation.csv`
  - `data/eval_entity_feasibility_unified_features/rf_feature_importance.csv`
  - `data/eval_entity_feasibility_unified_features/rf_group_importance.csv`
  - `data/eval_entity_feasibility_unified_features/xgb_feature_importance.csv`
  - `data/eval_entity_feasibility_unified_features/xgb_group_importance.csv`

  目前 first-cut 結果：
  - `format` 是最強 feature group
  - `context` 與 `graph` 次之
  - 拿掉 `format` 後，RF/XGB 都會掉到：
    - `accuracy = 0.646154`
    - `f1 = 0.711297`
    - `roc_auc = 0.662638`
  - 拿掉 `context` 會有可見下降
  - `role / local / opcode` 目前影響較小

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

- `eval_in_domain_xgb.py`
  比較 RandomForest 與 XGBoost 的 in-domain 表現。

- `tune_xgb_in_domain.py`
  掃一批 XGBoost 參數組合，輸出 `summary.json` 與 `candidates.csv`。

- `plot_benchmark_bar_png.py`
  把 benchmark `compare.csv` 畫成 `pred vs actual` 長條圖，可調大字體，也可把數值直接標在圖上。

- `plot_benchmark_compare_png.py`
  把 benchmark `compare.csv` 畫成散點圖，方便看 `pred` 與 `actual` 是否落在同一量級。

## Benchmark 驗證

- `eval_atax_benchmark.py`
  用真正的 PolyBench `atax.c` 產生 benchmark 真值，再呼叫 `predict_ir_errors.py` 對同一份 LLVM IR 做推論，最後輸出 `actual/pred/compare` 檔。

- `eval_bicg_benchmark.py`
  做法同上，對 `bicg` 產生 benchmark 真值與 prediction 對照。

這兩支都會另外輸出 `format_features.json`，讓 benchmark 推論也能真的吃到 quantization-aware features。

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
