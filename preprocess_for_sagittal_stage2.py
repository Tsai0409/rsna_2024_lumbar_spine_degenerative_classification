# preprocess_for_sagittal_stage2.py
import pandas as pd
import numpy as np

WORKING_DIR="/kaggle/working/duplicate"  # 我加

def sigmoid(x):
    return 1/(1 + np.exp(-x))

for fold in range(5):
    targets = ['l1_spinal', 'l2_spinal', 'l3_spinal', 'l4_spinal', 'l5_spinal']
    pred_cols = [f'pred_{c}' for c in targets]
    configs = [
        'rsna_sagittal_level_cl_spinal_v1',
    ]
    preds = []
    for config in configs:
        # test = pd.read_csv(f'results/{config}/test_fold{fold}.csv')
        test = pd.read_csv(f'{WORKING_DIR}/{config}/test_fold{fold}.csv')
        preds.append(test[pred_cols].values)
    test[pred_cols] = np.mean(preds, 0)
    test[pred_cols] = sigmoid(test[pred_cols]).astype(float)
    spinal = test.copy()

    targets = ['l1_right_neural', 'l2_right_neural', 'l3_right_neural', 'l4_right_neural', 'l5_right_neural', 'l1_left_neural', 'l2_left_neural', 'l3_left_neural', 'l4_left_neural', 'l5_left_neural']
    pred_cols = [f'pred_{c}' for c in targets]
    configs = [
        'rsna_sagittal_level_cl_nfn_v1',
    ]

    preds = []
    for config in configs:
        # test = pd.read_csv(f'results/{config}/test_fold{fold}.csv')
        test = pd.read_csv(f'{WORKING_DIR}/{config}/test_fold{fold}.csv')
        preds.append(test[pred_cols].values)
    test[pred_cols] = np.mean(preds, 0)
    test[pred_cols] = sigmoid(test[pred_cols]).astype(float)
    nfn = test.copy()
    df = spinal.merge(nfn[['path']+pred_cols], on='path')
    # fold_df = pd.read_csv('input/train_with_fold.csv').drop_duplicates('study_id')[['fold', 'study_id']]
    fold_df = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_4/train_with_fold.csv').drop_duplicates('study_id')[['fold', 'study_id']]
    # df.merge(fold_df, on='study_id').to_csv(f'input/train_for_sagittal_level_cl_v1_for_train_spinal_nfn_fold{fold}.csv', index=False)
    df.merge(fold_df, on='study_id').to_csv(f'{WORKING_DIR}/train_for_sagittal_level_cl_v1_for_train_spinal_nfn_fold{fold}.csv', index=False)
    