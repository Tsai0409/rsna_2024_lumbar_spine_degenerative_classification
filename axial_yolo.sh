# axial_yolo.sh
#!/bin/bash
set +e  # 遇到錯誤時繼續執行

# 設定工作目錄與腳本路徑
WORKING_DIR="/kaggle/working/duplicate"
PREPROCESS_SCRIPT="$WORKING_DIR/preprocess_for_axial_yolo.py"
TRAIN_SCRIPT="$WORKING_DIR/yolox_train_one_fold.py"

# 設定 YOLOX 路徑
export PYTHONPATH=/kaggle/working/duplicate/src/YOLOX:$PYTHONPATH  # 新增 

# 配置名稱與摺疊數
# configs=("rsna_axial_all_images_left_yolox_x" "rsna_axial_all_images_right_yolox_x")
configs=("rsna_axial_all_images_left_yolox_x")
# configs=("rsna_axial_all_images_right_yolox_x")
folds=(0 1)

# 檢查所需的腳本是否存在
if [[ ! -f $PREPROCESS_SCRIPT || ! -f $TRAIN_SCRIPT ]]; then
    echo "Error: Missing required scripts in $WORKING_DIR"
    exit 1
fi

# 執行預處理
# cmd="python $PREPROCESS_SCRIPT"
# echo "Executing: $cmd"
# if ! eval $cmd; then
#     echo "Error: Preprocessing failed."
#     exit 1
# fi

# 遍歷配置和摺疊數進行訓練
for config in "${configs[@]}"; do
    for fold in "${folds[@]}"; do
        cmd="python $TRAIN_SCRIPT -c $config -f $fold"
        echo "Executing: $cmd"
        if ! eval $cmd; then
            echo "Error: Training failed for config $config fold $fold."
            continue  # 當前配置失敗則跳過
        fi
        echo "----------------------------------------"
    done
done

echo "axial_yolo.sh completed successfully!"