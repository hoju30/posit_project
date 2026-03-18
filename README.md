# posit_test 專案說明

以多種 posit 格式與 fp32 執行常見運算（dot/sum/axpy_sum/relu_sum/l1_norm/sum_squares/axpby_sum/matvec/prefix_sum），量測誤差並用 ir2vec 特徵訓練模型，預測適合的數值格式或直接預測各格式相對 fp64 的誤差。格式上限為 posit 8/16/32 且 es≤2。

## 專案結構
- `src/`：C++ 原始碼（`main_*` 產生資料集；`kernel.*` 基本運算）。
- `bin/`：編譯後可執行檔。
- `data/`：資料集與中間結果（`*_dataset.csv`、`ir2vec_features.npz`、`ml_dataset_*.npz` 等）。
- `models/`：訓練好的模型（誤差回歸 `ir2vec_error_predictor.joblib`）。
- `scripts/`：資料處理、訓練、推論、繪圖腳本（常用入口）。
- `scripts/tools/`：實驗/串接工具（MLIR plan、LLVM IR sanitize、segment↔loop_id 對應）。
- `ir/`：各 kernel 的 LLVM IR（`.ll/.bc`）。
- `external/`：第三方庫（universal）。

## 模型
- 誤差回歸：IR2Vec +（vec_len/rows/cols）+（dot 的 x/y 統計）→ 各格式（posit 8/16/32 es≤2、fp32）的相對誤差（vs fp64），`models/ir2vec_error_predictor.joblib`（推論 `scripts/predict_ir_errors.py`）。
- 皆為 sklearn RandomForest。

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

./bin/dot_run --random-scale --samples 2000   # dot 會同時輸出 x/y 的 mean/std/min/max
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
python scripts/predict_ir_errors.py ir/dot.ll --vec-len 128 \
  --x-mean 0 --x-std 1 --x-p1 -3 --x-p50 0 --x-p99 3 \
  --y-mean 0 --y-std 10 --y-p1 -30 --y-p50 0 --y-p99 30 \
  --save-csv data/dot_error_preds.csv
python scripts/plot_bar.py --csv data/dot_error_preds.csv --out-prefix data/dot_error_bar
```

## 注意事項
- ir2vec 需能解析生成的 IR，請用 clang-16 並加 `-Xclang -no-opaque-pointers`。
- C++ 使用 C++20（universal 依賴 `std::bit_cast` 等特性）。
- Python 請用專案的 venv（例如 `.venv/bin/python ...` 或 `source .venv/bin/activate`），避免缺套件（如 pandas）。
