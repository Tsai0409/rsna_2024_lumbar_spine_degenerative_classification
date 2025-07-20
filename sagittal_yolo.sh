# sagittal_yolo.sh
#!/bin/bash
set +e  # 允許遇到錯誤時繼續執行，但我們會手動檢查每一步的返回值

# 下載與解壓縮資料集
# INPUT_DIR="./input"
# INPUT_DIR="/kaggle/working/duplicate"
# if [[ ! -d "$INPUT_DIR" ]]; then
#     echo "Error: 找不到 input 資料夾"
#     exit 1
# fi

# cd "$INPUT_DIR"
# cmd="kaggle datasets download -d yujiariyasu/bartley-coords-rsna-improved-csv"
# echo "Executing: $cmd"
# if ! eval $cmd; then
#     echo "Error: 資料集下載失敗。"
#     exit 1
# fi

# cmd="unzip bartley-coords-rsna-improved-csv.zip"
# echo "Executing: $cmd"
# if ! eval $cmd; then
#     echo "Error: 解壓縮資料集失敗。"
#     exit 1
# fi
# cd ..

# 設定 YOLOX 路徑
export PYTHONPATH=/kaggle/working/duplicate/src/YOLOX:$PYTHONPATH  # 新增 

# 設定工作目錄與必要腳本路徑
WORKING_DIR="/kaggle/working/duplicate"
PREPROCESS_SCRIPT="$WORKING_DIR/preprocess_for_sagittal_yolo.py"
TRAIN_SCRIPT="$WORKING_DIR/yolox_train_one_fold.py"

# 檢查所需腳本是否存在
if [[ ! -f $PREPROCESS_SCRIPT || ! -f $TRAIN_SCRIPT ]]; then
    echo "Error: 缺少必要的腳本，請確認 $WORKING_DIR 中有 preprocess_for_sagittal_yolo.py 與 yolox_train_one_fold.py"
    exit 1
fi

# 執行預處理 (finish)
# cmd="python $PREPROCESS_SCRIPT"
# echo "Executing: $cmd"
# if ! eval $cmd; then
#     echo "Error: 預處理失敗。"
#     exit 1
# fi

# 配置與摺疊設定
configs=("rsna_10classes_yolox_x")
# folds=(0 1 2 3 4)
folds=(1)

# 遍歷每個配置與摺疊進行訓練
for config in "${configs[@]}"; do
    for fold in "${folds[@]}"; do
        cmd="python $TRAIN_SCRIPT -c $config -f $fold"
        echo "Executing: $cmd"
        if ! eval $cmd; then
            echo "Error: config $config fold $fold 訓練失敗，略過該組合。"
            continue
        fi
        echo "----------------------------------------"
    done
done

echo "sagittal_yolo.sh completed successfully!"