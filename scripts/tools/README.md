# scripts/tools

放一些「串接 / 實驗」用的小工具（不一定每次都會跑）。

## 檔案

- `gen_mlir_loop_plan.py`
  - 從 MLIR（跑 `--posit-assign-loop-id` 之後）抽出所有 `posit.loop_id`，產生 pass 可吃的 `plan.json`
  - 可用 `--pred-json` / `--pred-csv` 把 model 輸出的每個 datatype 誤差（`pred_rel_err`）填進去

- `sanitize_llvm_ir_for_llvm16.py`
  - 將「較新 LLVM 產生的 .ll」做最小限度的語法修補，讓 `opt-16/llvm-extract-16/ir2vec` 仍能解析

- `build_loop_plan_from_segments.py`
  - 用「順序對齊」的方式，把 `predict_ir_errors_by_loops.py --save-json` 的 segments 映射到 MLIR 的 `loop_id`
  - 只適合 demo/loop 數量一致且順序不會跑掉的情境

- `build_loop_plan_from_marked_segments.py`
  - 建議使用：依 `__posit_loop_marker(i64 hash)` 做 segment↔loop_id 對應（比較穩健）
  - 需要先在 MLIR 跑 `--posit-insert-loop-marker`

