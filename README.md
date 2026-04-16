# posit_test 專案說明

這個專案用常見 kernel（dot/sum/axpy_sum/relu_sum/l1_norm/sum_squares/axpby_sum/matvec/prefix_sum）的數值實驗資料，搭配 LLVM IR 的 IR2Vec 特徵，訓練模型去預測各種 posit 格式的相對誤差。目前格式範圍是 `posit(8/16/32, es=0..2)`；主線是 `per-format regression`，推論時會對 9 個 posit type 逐一輸出誤差表。

## 專案結構
- `src/`：C++ 原始碼（`main_*` 產生資料集；`kernel.*` 基本運算）。
- `bin/`：編譯後可執行檔。
- `data/`：資料集與中間結果（`*_dataset.csv`、`ir2vec_features.npz`、`ml_dataset_*.npz` 等）。
- `models/`：訓練好的模型（RandomForest：`ir2vec_error_predictor.joblib`；XGBoost：`ir2vec_error_predictor_xgb.joblib`）。
- `scripts/`：資料處理、訓練、推論、繪圖腳本（常用入口）。
- `scripts/tools/`：實驗/串接工具（MLIR plan、LLVM IR sanitize、segment↔loop_id 對應）。
- `ir/`：各 kernel 的 LLVM IR（`.ll/.bc`）。
- `external/`：第三方庫（universal）。

## 模型
- 主線模型：
  - `models/ir2vec_error_predictor.joblib`：RandomForest baseline。
  - `models/ir2vec_error_predictor_xgb.joblib`：XGBoost 主線模型。
- 目前兩者都是 `per-format regression + log-target + selective calibration`。
- XGBoost 目前採用的較佳參數是：
  - `n_estimators=600`
  - `max_depth=4`
  - `learning_rate=0.02`
  - `subsample=0.8`
  - `colsample_bytree=0.8`
  - `reg_lambda=4.0`
  - `min_child_weight=5.0`

### Feature（簡述）
- 目前正式輸入維度是 `359 = 300 IR2Vec + 43 shared + 16 format-aware`。
- `shared features`：scalar（`vec_len/rows/cols/alpha/beta`）+ raw stats（`mean/std/min/max/abs_max/skewness/excess_kurtosis/p01/p50/p99/near_zero_ratio/pos_ratio/neg_ratio`）+ derived stats（`log10_std/upper_tail/lower_tail/standardized_mean/center_decade/span_decades`）。
- `format-aware features`：`current-es excess`（`x/y_upper_excess_current_es`、`x/y_lower_excess_current_es`）+ quant features（`oor_ratio/clip_high_ratio/clip_low_ratio/rel_qerr_mean/zero_after_quant_ratio/unique_ratio`）。
- raw stats 與 quant features 的 C++ 實作在 [stats.hpp](/home/hoju/test/posit_test/src/stats.hpp)；derived/current-es feature 的 Python 實作在 [error_feature_utils.py](/home/hoju/test/posit_test/scripts/error_feature_utils.py)；組成最終訓練向量的位置在 [build_ml_dataset_ir_errors.py](/home/hoju/test/posit_test/scripts/build_ml_dataset_ir_errors.py)。

### Variable-level mixed-precision（草稿）
- 目前主線仍是 loop / app-level error prediction。
- whole-app source corpus 目前可由 [generate_source_apps.py](/home/hoju/test/posit_test/scripts/generate_source_apps.py) 批量產生：
  - `8 families`
  - `25 variants / family`
  - 共 `200 apps`
- 新增 [extract_entity_features.py](/home/hoju/test/posit_test/scripts/extract_entity_features.py) 作為 variable-level 路線的第一步：直接對整個 app 的 LLVM IR 抽 entity。
- 目前先抓三類 entity：
  - `accumulator`
  - `loop_carried_state`
  - `reduction_intermediate`
- 目前先輸出這些 feature：
  - entity-local：`role / def_opcode / use_count / is_loop_carried / is_output_related`
  - graph-derived：`fanout / reduction_depth / distance_to_output / same_loop_neighbor_count / num_arith_users`
- 其中 `same_loop_neighbor_count` 在沒有顯式 loop analysis 前，先用 same-function def-use neighbors 近似。
- entity label schema 目前已改成 whole-app binary classification 命名：
  - `entity_id`
  - `format`
  - `actual_whole_app_rel_err`
  - `is_feasible_under_tol`
  - `margin_to_tol`
- 目前已補上 generated-app accumulator evaluator：
  - [compile_generated_apps.py](/home/hoju/test/posit_test/scripts/compile_generated_apps.py)：把 generated apps 編成 LLVM IR，並產生 entity feature / skeleton label CSV。
  - [fill_entity_labels.py](/home/hoju/test/posit_test/scripts/fill_entity_labels.py)：對 `generated_apps` 中「只有一個 accumulator entity」的 app，回填
    - `actual_whole_app_rel_err`
    - `is_feasible_under_tol`
    - `margin_to_tol`
  - [generated_app_accumulator_eval.cpp](/home/hoju/test/posit_test/src/generated_app_accumulator_eval.cpp)：目前 actual label 的 source-level evaluator，先支援 accumulator-only。
  - 目前全量 `200 apps` 中，這條 first cut 已能直接回填 `150 apps`；`ema_recurrence` 與 `matvec_like` 仍先保留 skeleton。
  - [train_entity_feasibility_model.py](/home/hoju/test/posit_test/scripts/train_entity_feasibility_model.py)：用合併後的 accumulator label dataset 訓練第一版 binary feasibility classifier。
  - [train_entity_feasibility_model_xgb.py](/home/hoju/test/posit_test/scripts/train_entity_feasibility_model_xgb.py)：XGBoost 版本的 accumulator feasibility classifier。
  - 現在三類 entity 都已有獨立 dataset：
    - `data/generated_entity_label_dataset_accum.csv`
    - `data/generated_entity_label_dataset_lcs.csv`
    - `data/generated_entity_label_dataset_ri.csv`
  - 若要做 unified entity model，可用 [merge_role_entity_datasets.py](/home/hoju/test/posit_test/scripts/merge_role_entity_datasets.py) 合成：
    - `data/generated_entity_label_dataset_all.csv`
  - 目前 unified dataset 一共有 `2925` rows，已訓練：
    - `models/entity_feasibility_all_rf.joblib`
    - `models/entity_feasibility_all_xgb.joblib`
  - unified holdout：
    - RF：`accuracy=0.945299`, `f1=0.952941`, `roc_auc=0.991090`
    - XGB：`accuracy=0.945299`, `f1=0.952941`, `roc_auc=0.990035`
  - 也已補上 unified feature importance / ablation：
    - [eval_entity_feasibility_unified_features.py](/home/hoju/test/posit_test/scripts/eval_entity_feasibility_unified_features.py)
    - 結果在：
      - `data/eval_entity_feasibility_unified_features/`
    - 目前觀察：
      - `format` 是最強訊號
      - `context` 與 `graph` 次之
      - 拿掉 `format` 後，模型表現會明顯崩落

## 流程（啟用 venv，於專案根目錄）

### 1) 生成資料集（C++）
```bash
g++ -std=c++20 -O2 src/main_dot.cpp src/kernel.cpp -Iexternal/universal/include/sw -o bin/dot_run
g++ -std=c++20 -O2 src/main_sum.cpp src/kernel.cpp -Iexternal/universal/include/sw -o bin/sum_run
g++ -std=c++20 -O2 src/main_axpy_sum.cpp src/kernel.cpp -Iexternal/universal/include/sw -o bin/axpy_run
g++ -std=c++20 -O2 src/main_relu_sum.cpp src/kernel.cpp -Iexternal/universal/include/sw -o bin/relu_run
g++ -std=c++20 -O2 src/main_l1_norm.cpp src/kernel.cpp -Iexternal/universal/include/sw -o bin/l1_norm_run
g++ -std=c++20 -O2 src/main_sum_squares.cpp src/kernel.cpp -Iexternal/universal/include/sw -o bin/sum_squares_run
g++ -std=c++20 -O2 src/main_axpby_sum.cpp src/kernel.cpp -Iexternal/universal/include/sw -o bin/axpby_sum_run
g++ -std=c++20 -O2 src/main_matvec.cpp src/kernel.cpp -Iexternal/universal/include/sw -o bin/matvec_run
g++ -std=c++20 -O2 src/main_prefix_sum.cpp src/kernel.cpp -Iexternal/universal/include/sw -o bin/prefix_sum_run

./bin/dot_run --random-scale --samples 2000   # dot 會同時輸出完整 x/y raw stats
./bin/sum_run
./bin/axpy_run
./bin/relu_run
./bin/l1_norm_run
./bin/sum_squares_run
./bin/axpby_sum_run
./bin/matvec_run
./bin/prefix_sum_run
```

### 2) 產生 LLVM IR（clang-16，關閉 opaque pointers）
```bash
# 推薦：用純 kernel 的 IR（避免把 RNG/IO/資料生成混進 ir2vec 特徵）：
bash scripts/emit_kernel_ir.sh

# 或手動逐一產生：
clang-16 -std=c++20 -O2 -S -emit-llvm -Xclang -no-opaque-pointers src/ir_kernels/dot.cpp         -o ir/dot.ll
clang-16 -std=c++20 -O2 -S -emit-llvm -Xclang -no-opaque-pointers src/ir_kernels/sum.cpp         -o ir/sum.ll
clang-16 -std=c++20 -O2 -S -emit-llvm -Xclang -no-opaque-pointers src/ir_kernels/axpy_sum.cpp    -o ir/axpy_sum.ll
clang-16 -std=c++20 -O2 -S -emit-llvm -Xclang -no-opaque-pointers src/ir_kernels/relu_sum.cpp    -o ir/relu_sum.ll
clang-16 -std=c++20 -O2 -S -emit-llvm -Xclang -no-opaque-pointers src/ir_kernels/l1_norm.cpp     -o ir/l1_norm.ll
clang-16 -std=c++20 -O2 -S -emit-llvm -Xclang -no-opaque-pointers src/ir_kernels/sum_squares.cpp -o ir/sum_squares.ll
clang-16 -std=c++20 -O2 -S -emit-llvm -Xclang -no-opaque-pointers src/ir_kernels/axpby_sum.cpp   -o ir/axpby_sum.ll
clang-16 -std=c++20 -O2 -S -emit-llvm -Xclang -no-opaque-pointers src/ir_kernels/matvec.cpp      -o ir/matvec.ll
clang-16 -std=c++20 -O2 -S -emit-llvm -Xclang -no-opaque-pointers src/ir_kernels/prefix_sum.cpp  -o ir/prefix_sum.ll
```

### 3) ir2vec 特徵
```bash
python scripts/extract_ir2vec_features.py
```
輸出：`data/ir2vec_features.npz`。

### 4) 產生標註（分類用）
```bash
（此 repo 目前以「誤差回歸」為主；分類管線已移除/不再維護。）
```

### 5) 誤差回歸資料集與模型
```bash
python scripts/build_ml_dataset_ir_errors.py   # → data/ml_dataset_ir_errors.npz
python scripts/train_ir_error_model.py         # → models/ir2vec_error_predictor.joblib
python scripts/train_ir_error_model_xgb.py     # → models/ir2vec_error_predictor_xgb.joblib
python scripts/predict_ir_errors.py ir/dot.ll --vec-len 128 \
  --x-mean 0 --x-std 1 --x-min -3 --x-max 3 --x-abs-max 3 --x-skewness 0 --x-excess-kurtosis 0 --x-p01 -2.3 --x-p50 0 --x-p99 2.3 --x-near-zero-ratio 0.01 --x-pos-ratio 0.5 --x-neg-ratio 0.5 \
  --y-mean 0 --y-std 10 --y-min -30 --y-max 30 --y-abs-max 30 --y-skewness 0 --y-excess-kurtosis 0 --y-p01 -23 --y-p50 0 --y-p99 23 --y-near-zero-ratio 0.01 --y-pos-ratio 0.5 --y-neg-ratio 0.5 \
  --save-csv data/dot_error_preds.csv
python scripts/plot_bar.py --csv data/dot_error_preds.csv --out-prefix data/dot_error_bar
```

## 注意事項
- ir2vec 需能解析生成的 IR，請用 clang-16 並加 `-Xclang -no-opaque-pointers`。
- C++ 使用 C++20（universal 依賴 `std::bit_cast` 等特性）。
- Python 請用專案的 venv（例如 `.venv/bin/python ...` 或 `source .venv/bin/activate`），避免缺套件（如 pandas）。
- benchmark 驗證目前有 `atax`、`bicg`：
  - `scripts/eval_atax_benchmark.py`
  - `scripts/eval_bicg_benchmark.py`
  會輸出 `actual/pred/compare.json` 與 `compare.csv`，可再搭配 `scripts/plot_benchmark_bar_png.py` 畫 `pred vs actual` 長條圖。
