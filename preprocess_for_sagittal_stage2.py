# preprocess_for_sagittal_stage2.py
# !python {Target_path}/preprocess_for_sagittal_stage2.py
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
    # spinal.to_csv('spinal.csv', index=False)

    
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
    # nfn.to_csv('nfn.csv', index=False)

    df = spinal.merge(nfn[['path']+pred_cols], on='path')
    df.to_csv('df.csv', index=False)
    # fold_df = pd.read_csv('input/train_with_fold.csv').drop_duplicates('study_id')[['fold', 'study_id']]
    # fold_df = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_4/train_with_fold.csv').drop_duplicates('study_id')[['fold', 'study_id']]  # 篩選 DataFrame 只保留 fold 和 study_id 這兩個欄位，丟棄其他欄位
    fold_df = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_holdout_4/train_with_fold_holdout_test.csv').drop_duplicates('study_id')[['fold', 'study_id']]
    # fold_df = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_holdout_4/train_with_fold_holdout.csv', low_memory=False).drop_duplicates('study_id')[['fold', 'study_id']]
    # fold_df.to_csv('fold_df.csv', index=False)

    # df.merge(fold_df, on='study_id').to_csv(f'input/train_for_sagittal_level_cl_v1_for_train_spinal_nfn_fold{fold}.csv', index=False)
    # df.merge(fold_df, on='study_id')  # .to_csv(f'train_for_sagittal_level_cl_v1_for_train_spinal_nfn_fold{fold}.csv', index=False)  # 這個產生的用意是什麼？
    label_df = df.merge(fold_df, on='study_id')


    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    # 假設 df 是你 merge 完的資料（已經有 study_id），而 label_df 是標註檔

    # 1. 選出標籤欄位中 "Severe" 太少的
    label_features = [
        # Spinal canal stenosis
        'spinal_canal_stenosis_l1_l2',
        'spinal_canal_stenosis_l2_l3',
        'spinal_canal_stenosis_l3_l4',
        'spinal_canal_stenosis_l4_l5',
        'spinal_canal_stenosis_l5_s1',

        # Left neural foraminal narrowing
        'left_neural_foraminal_narrowing_l1_l2',
        'left_neural_foraminal_narrowing_l2_l3',
        'left_neural_foraminal_narrowing_l3_l4',
        'left_neural_foraminal_narrowing_l4_l5',
        'left_neural_foraminal_narrowing_l5_s1',

        # Right neural foraminal narrowing
        'right_neural_foraminal_narrowing_l1_l2',
        'right_neural_foraminal_narrowing_l2_l3',
        'right_neural_foraminal_narrowing_l3_l4',
        'right_neural_foraminal_narrowing_l4_l5',
        'right_neural_foraminal_narrowing_l5_s1',
    ]

    # 建立一個 set 來避免重複
    label_features_set = set(label_features)

    # 根據條件補充其他欄位
    for c in list(label_df):
        if c == 'study_id' or c in label_features_set:
            continue
        if (label_df[c] == 'Severe').sum() < 30:
            label_features.append(c)
            label_features_set.add(c)  # 確保後續不會重複加

    # 2. 對應欄位做 One-Hot（缺值補 Normal/Mild）
    one_hot_labels = label_df[label_features].fillna('Normal/Mild').values

    # 3. 建立 fold 列
    label_df['fold'] = -1
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
    for fold, (train_idx, val_idx) in enumerate(mskf.split(one_hot_labels, one_hot_labels)):
        label_df.loc[val_idx, 'fold'] = fold

    # 4. 把 fold merge 回你原本的 df
    df = df.drop(columns=[c for c in df.columns if c.startswith('fold')], errors='ignore')  # 移除 fold_x/fold_y

    # df = df.merge(label_df[['study_id', 'fold']], on='study_id', how='left')
    label_df_unique = label_df.drop_duplicates('study_id')  # 保證不會多對多
    df = df.merge(label_df_unique[['study_id', 'fold']], on='study_id', how='left')

    df['fold'] = df['fold'].astype(int)

    # 5. 輸出 CSV
    df.to_csv(f'train_for_sagittal_level_cl_v1_for_train_spinal_nfn_fold0.csv', index=False)

print('preprocess_for_sagittal_stage2.py finish')