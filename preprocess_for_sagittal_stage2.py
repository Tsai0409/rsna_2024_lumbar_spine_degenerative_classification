# preprocess_for_sagittal_stage2.py
import pandas as pd
import numpy as np

WORKING_DIR="/kaggle/working/duplicate"  

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# for fold in range(5):
for fold in range(1):
    targets = ['l1_spinal', 'l2_spinal', 'l3_spinal', 'l4_spinal', 'l5_spinal']
    pred_cols = [f'pred_{c}' for c in targets]
    configs = [
        'rsna_sagittal_level_cl_spinal_v1',
    ]
    preds = []
    for config in configs:
        # test = pd.read_csv(f'results/{config}/test_fold{fold}.csv')
        test = pd.read_csv(f'{WORKING_DIR}/ckpt/{config}/test_fold{fold}.csv')  # /kaggle/working/duplicate/ckpt/rsna_sagittal_level_cl_spinal_v1/test_fold0.csv (spinal test dataset)
        preds.append(test[pred_cols].values)  # 取出特定欄位 pred_cols（這些欄位代表各個標籤的預測結果），並使用 .values 轉換為 numpy 陣列
    test[pred_cols] = np.mean(preds, 0)  # 將 test_fold0、test_fold1 各個欄位取平均；對所有 config 的預測結果按 axis=0 求均值，這表示對同一筆資料，不同模型（或不同設定）的預測結果取平均
    test[pred_cols] = sigmoid(test[pred_cols]).astype(float)  # 轉換為 0-1 的數字(做正規化)
    spinal = test.copy()

    targets = ['l1_right_neural', 'l2_right_neural', 'l3_right_neural', 'l4_right_neural', 'l5_right_neural', 'l1_left_neural', 'l2_left_neural', 'l3_left_neural', 'l4_left_neural', 'l5_left_neural']
    pred_cols = [f'pred_{c}' for c in targets]
    configs = [
        'rsna_sagittal_level_cl_nfn_v1',
    ]

    preds = []
    for config in configs:
        # test = pd.read_csv(f'results/{config}/test_fold{fold}.csv')
        test = pd.read_csv(f'{WORKING_DIR}/ckpt/{config}/test_fold{fold}.csv')
        preds.append(test[pred_cols].values)
    test[pred_cols] = np.mean(preds, 0)
    test[pred_cols] = sigmoid(test[pred_cols]).astype(float)
    nfn = test.copy()

    df = spinal.merge(nfn[['path']+pred_cols], on='path')
    # fold_df = pd.read_csv('input/train_with_fold.csv').drop_duplicates('study_id')[['fold', 'study_id']]
    # fold_df = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_4/train_with_fold.csv').drop_duplicates('study_id')[['fold', 'study_id']]  # 篩選 DataFrame 只保留 fold 和 study_id 這兩個欄位，丟棄其他欄位
    # fold_df = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_holdout_4/train_with_fold_holdout.csv').drop_duplicates('study_id')[['fold', 'study_id']]
    fold_df = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_holdout_4/train_with_fold_holdout.csv', low_memory=False).drop_duplicates('study_id')[['fold', 'study_id']]
    # df.merge(fold_df, on='study_id').to_csv(f'input/train_for_sagittal_level_cl_v1_for_train_spinal_nfn_fold{fold}.csv', index=False)
    df.merge(fold_df, on='study_id').to_csv('train_for_sagittal_level_cl_v1_for_train_spinal_nfn_fold{fold}.csv', index=False)  # 這個產生的用意是什麼？
    
print('preprocess_for_sagittal_stage2.py finish')