# Classification Pipeline（格式選擇）

此分支專注於從 IR 預測最適的數值格式（posit 8/16/32, es≤2 或 fp32）。

## 步驟
1. **C++ 產生資料**（共用 `bin/*.run`，輸出 `data/*_dataset.csv`）。
2. **產生 LLVM IR**（clang-16 `-no-opaque-pointers`，IR 放 `ir/*.ll`）。
   - 建議用純 kernel IR：`bash scripts/emit_kernel_ir.sh`
3. **抽取 ir2vec 特徵**  
   `python scripts/extract_ir2vec_features.py` → `data/ir2vec_features.npz`
4. **產生標註**  
   `python scripts/label_kernels_accuracy.py` → `data/kernel_labels_by_len_dist.csv`
5. **組合資料集**  
   `python scripts/build_ml_dataset_ir.py` → `data/ml_dataset_ir.npz`
6. **訓練分類模型**  
   `python scripts/train_ir_model.py` → `models/ir2vec_posit_selector.joblib`
7. **推論**  
   `python scripts/predict_ir_format.py ir/dot.ll`

## 輸出
- 模型：`models/ir2vec_posit_selector.joblib`
- 推論腳本：`scripts/predict_ir_format.py`

> 資料生成與 IR 產生與回歸管線共用，流程細節可參考專案根的 README。
