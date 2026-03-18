# Regression Pipeline（誤差預測）

此分支專注於從 LLVM IR 預測各格式（posit 8/16/32, es≤2、fp32）對 fp64 的相對誤差。

目前回歸模型的輸入特徵（X）：
- `ir2vec` 程式向量（300 維）
- 尺寸特徵：`vec_len/rows/cols`（3 維，會做縮放）
- 係數特徵：`alpha/beta`（2 維；axpy/axpby 用，其他 kernel 為 0）
- dot 專用的 runtime 統計特徵：推論端提供 `x/y` 的 `mean/std/p1/p50/p99`（其中 p1/p50/p99 建議用 `|x|` 的分位數），模型內轉成 24 維（x 12 + y 12）  
  - 尺度/形狀：`log10(std+eps)` + `upper_tail/lower_tail/standardized_mean`  
  - posit sweet spot：`center_decade=log10(p50+eps)`, `span_decades=log10(p99+eps)-log10(p1+eps)`  
    以及對 `es=0..2` 的 `upper_excess/lower_excess`
- CFG 彙總特徵（8 維）：從 loop segment 的 basic blocks / CFG 結構萃取（`bb_count/edge_count/avg_succ/branch_bb_frac/instr_log1p/has_cycle/cycle_edge_frac/largest_scc_frac`）

dist_type 不再以 one-hot 形式餵給模型；它最多只用在資料生成/切片分析。

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
6. **推論誤差**  
   `python scripts/predict_ir_errors.py ir/dot.ll --vec-len 128 --x-mean 0 --x-std 1 --x-p1 -3 --x-p50 0 --x-p99 3 --y-mean 0 --y-std 10 --y-p1 -30 --y-p50 0 --y-p99 30 --save-csv data/dot_error_preds.csv`
7. **繪圖（可選）**  
   `python scripts/plot_bar.py --csv data/dot_error_preds.csv --out-prefix data/dot_error_bar`

## 混合精度（per-loop）推論
當你輸入的是「大型 IR」且包含多個 loop，建議用 per-loop 的方式避免全域 IR2Vec 向量混到不相關的訊號：

1. **依 loop 切段**（會輸出 `manifest.json` + 多個 `.ll` segment）  
   `python scripts/split_loops_and_analyze_cfg.py path/to/input.ll --segments-dir ir/segments/input --no-cfg`
2. **逐段推論 + 產生混合精度決策（不改 IR，只輸出 plan）**  
   `python scripts/predict_ir_errors_by_loops.py path/to/input.ll --vec-len 128 --x-mean 0 --x-std 1 --x-p1 -3 --x-p50 0 --x-p99 3 --y-mean 0 --y-std 10 --y-p1 -30 --y-p50 0 --y-p99 30 --pick min_bits_under_tol --tol 1e-3`

輸出：
- `ir/segments/<stem>/predictions.json`：每個 loop segment 的 10 維誤差預測 + `chosen`（該段建議格式）
- 可加 `--save-csv data/<stem>_agg.csv` 產生聚合後的 `target,pred`，再用 `plot_bar.py` 畫圖

## CFG / Basic Block 分析（進階）
若你要在 per-loop segment 內做 basic block 等級的 CFG 分析（例如想用 SCC 把含 back-edge 的 CFG 壓縮成 DAG 再做圖上分析），可用：

- 對整份 IR（單檔）輸出 CFG JSON：  
  `python scripts/split_loops_and_analyze_cfg.py ir/dot.ll --segments-dir ir/segments/dot_ir --no-cfg`
  （若你要單檔 CFG JSON，請先把 `.ll` 當成一個 segment 來分析，或自行指定 segment 檔案）
- 對 loop segments（manifest.json）批次輸出每段 CFG + SCC：  
  `python scripts/split_loops_and_analyze_cfg.py ir/dot.ll --segments-dir ir/segments/dot_demo`

若你想把「split loops + CFG 分析」合併成一個指令（推薦）：
- `python scripts/split_loops_and_analyze_cfg.py ir/dot.ll --segments-dir ir/segments/dot_demo`
  - 會自動產生/重用 `manifest.json`，並輸出 `ir/segments/dot_demo/cfg/*cfg.json`
  - 可選輸出 8 維 CFG 彙總特徵（和回歸模型一致）：  
    `--save-cfg-features-csv data/dot_demo_cfg_feats.csv`

SCC 計算模式：
- 預設 `--scc-mode auto`：只有當 CFG 有 cycle/back-edge 時才會輸出 SCC（否則 `scc.computed=false`，並提供 `dag.topo_bb_order`）
- 強制永遠算 SCC：`--scc-mode always`
- 完全關閉：`--scc-mode off`

輸出：
- `cfg/index.json`：segment→cfg 檔案索引
- `cfg/*cfg.json`：每個 segment 的 basic blocks、edges、SCC components、condensation DAG（topo order）

## 輸出
- 模型：`models/ir2vec_error_predictor.joblib`
- 推論腳本：`scripts/predict_ir_errors.py`
- 其他：誤差長條圖 `dot_error_bar*.png`、預測 CSV。

> 資料生成與 IR 產生與分類管線共用，流程細節可參考專案根的 README。
