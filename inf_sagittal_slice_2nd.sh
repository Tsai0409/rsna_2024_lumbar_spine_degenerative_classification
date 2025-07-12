# inf_sagittal_slice_2nd.sh
#!/bin/bash
set +e  # 確保遇到錯誤時，讓腳本繼續執行

# 設定環境變數
WORKING_DIR="/kaggle/working/duplicate"
PREPROCESS_SCRIPT="$WORKING_DIR/preprocess_for_sagittal_stage2.py"
TRAIN_SCRIPT="$WORKING_DIR/train_one_fold.py"
PREDICT_SCRIPT="$WORKING_DIR/predict.py"

# 配置名稱和摺疊數
configs=("rsna_sagittal_cl")
# folds=(0 1 2 3 4)
folds=(1)

# 確保需要的腳本存在
if [[ ! -f $PREPROCESS_SCRIPT || ! -f $TRAIN_SCRIPT || ! -f $PREDICT_SCRIPT ]]; then
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

# 遍歷配置和摺疊數進行訓練與預測
for config in "${configs[@]}"
do
    for fold in "${folds[@]}"
    do
        # 執行訓練腳本
        cmd="python $TRAIN_SCRIPT -c $config -f $fold"
        echo "Executing: $cmd"
        if ! eval $cmd; then
            echo "Error: Training failed for config $config fold $fold."
            continue  # 跳過失敗的配置
        fi

        # 執行預測腳本
        # infcmd="python $PREDICT_SCRIPT -c $config -f $fold"
        # echo "Executing: $infcmd"
        # if ! eval $infcmd; then
        #     echo "Error: Prediction failed for config $config fold $fold."
        #     continue  # 跳過失敗的配置
        # fi

        echo "----------------------------------------"
    done
done

echo "inf_sagittal_slice_2nd.sh completed successfully!"