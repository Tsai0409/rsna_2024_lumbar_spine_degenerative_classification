# sagittal_classification.sh
#!/bin/bash

# 設定環境變數
WORKING_DIR="/kaggle/working/duplicate"
PREPROCESS_SCRIPT="$WORKING_DIR/preprocess_for_sagittal_classification.py"
TRAIN_SCRIPT="$WORKING_DIR/train_one_fold.py"
PREDICT_SCRIPT="$WORKING_DIR/predict.py"

# 執行預處理 (finish)
cmd="python $PREPROCESS_SCRIPT"
echo "Executing: $cmd"
if ! eval $cmd; then
    echo "Error: Preprocessing failed."
    exit 1
fi

# 設置 configs 和 folds 變數
configs=(
    "rsna_saggital_mil_spinal_crop_x03_y05_with_valid" 
    "rsna_saggital_mil_spinal_crop_x03_y07_with_valid" 

    "rsna_saggital_mil_ss_crop_x03_y05_96_with_valid" 
    "rsna_saggital_mil_ss_crop_x03_y07_96_with_valid" 
    "rsna_saggital_mil_ss_crop_x03_y2_96_with_valid" 
    "rsna_saggital_mil_ss_crop_x1_y07_96_with_valid" 
    
    "rsna_saggital_mil_nfn_crop_x07_y1_v2_with_valid" 
    "rsna_saggital_mil_nfn_crop_x15_y1_v2_with_valid" 
    "rsna_saggital_mil_nfn_crop_x03_y1_v2_with_valid" 
    "rsna_saggital_mil_nfn_crop_x05_y05_v2_with_valid"
)
# folds=(0 1 2 3 4)
folds=(1)

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
            continue  # 跳過失敗的配置，繼續執行其他
        fi

        # 執行預測腳本
        infcmd="python $PREDICT_SCRIPT -c $config -f $fold"
        echo "Executing: $infcmd"
        if ! eval $infcmd; then
            echo "Error: Prediction failed for config $config fold $fold."
            continue  # 跳過失敗的配置，繼續執行其他
        fi

        echo "----------------------------------------"
    done
done

echo "Script completed successfully!"
