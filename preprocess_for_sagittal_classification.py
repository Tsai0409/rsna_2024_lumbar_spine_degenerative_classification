# preprocess_for_sagittal_classification.py
import pandas as pd
import numpy as np
from tqdm import tqdm

# kaggle input
DATA_KAGGLE_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"

# 設定環境變數
WORKING_DIR="/kaggle/working/duplicate"

# 找為什麼 training DataFrame 的資料為空？
# tr -> train_one_fold.csv 有 fold0-4 的資訊
# oof -> 合併 /results/rsna_10classes_yolox_x/oof_fold0.csv、/results/rsna_10classes_yolox_x/oof_fold1.csv 有 fold0-1 的資訊
# test -> wbf 沒有 fold 資訊

config = 'rsna_10classes_yolox_x'
box_cols = ['x_min', 'y_min', 'x_max', 'y_max']
# tr = pd.read_csv('input/train_with_fold.csv')
tr = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_4/train_with_fold.csv')
# oof = pd.concat([pd.read_csv(f'results/{config}/oof_fold{fold}.csv') for fold in range(5)])
oof = pd.concat([pd.read_csv(f'{WORKING_DIR}/results/{config}/oof_fold{fold}.csv') for fold in range(2)])
oof.to_csv('/kaggle/working/oof.csv')  # 我加
# test = pd.read_csv(f'results/wbf/{config}.csv')
test = pd.read_csv(f'{WORKING_DIR}/results/wbf/{config}.csv')
test['study_id'] = test.path.apply(lambda x: int(x.split('/')[-1].split('___')[0]))
test['series_id'] = test.path.apply(lambda x: int(x.split('/')[-1].split('___')[1]))
test = test[~test['study_id'].isin(oof.study_id)]  # 將在 oof 中的 study_id 全部移除
t2_ids = tr[tr.series_description_y == 'Sagittal T2/STIR'].series_id
test = test[test['series_id'].isin(t2_ids)]
test.to_csv('/kaggle/working/test.csv')  # 我加

dfs = []
for i, idf in test.groupby('study_id'):
    assert idf.series_id.nunique()==1
for i, idf in oof.groupby('study_id'):
    assert idf.series_id.nunique()==1

box_df = pd.concat([oof, test])
for id, name in oof.drop_duplicates('class_id')[['class_id', 'class_name']].values:
    box_df.loc[box_df.class_id==id, 'class_name'] = name

dfs = []
for id, idf in tqdm(box_df[['study_id', 'conf', 'class_id', 'class_name']+box_cols].groupby(['study_id', 'class_id'])):
    idf = idf[idf.conf==idf.conf.max()].iloc[:1]
    dfs.append(idf)
box_df = pd.concat(dfs)

error_dfs = []
for id, idf in box_df.groupby('study_id'):
    if len(idf) != 10:
        error_dfs.append(idf)
    assert len(idf)<=10

box_df['level'] = box_df.class_name.apply(lambda x: x.split('_')[0])
box_df['lr'] = box_df.class_name.apply(lambda x: x.split('_')[1])
box_df.to_csv('/kaggle/working/box_df.csv')  # 我加

rolling = 5
range_n = 2

# spinal
dfs = []
# df_path = 'results/rsna_sagittal_cl/oof.csv'
df_path = f'{WORKING_DIR}/ckpt/rsna_sagittal_cl/oof.csv'
df = pd.read_csv(df_path)
# df['path'] = f'input/sagittal_all_images/' + df.study_id.astype(str) + '___' + df.instance_number.astype(str) + '.png'
df['path'] = f'/kaggle/temp/sagittal_all_images/' + df.study_id.astype(str) + '___' + df.instance_number.astype(str) + '.png'
for id, idf in df.groupby('series_id'):
    idf = idf.sort_values(['x_pos', 'instance_number'])
    idf = idf.drop_duplicates('x_pos')
    idf[f'pred_spinal_rolling'] = idf[f'pred_spinal'].rolling(rolling, center=True).mean()    

    path_fit_xy = idf[idf['pred_spinal']==idf['pred_spinal'].max()].path.values[0]
    
    col = 'pred_spinal_rolling'
    n = idf[idf[col]==idf[col].max()].instance_number.values[0]

    ldf = idf[(idf.instance_number >= n-range_n) & (idf.instance_number <= n+range_n)]
    l_paths = ['nan'] * (1+range_n*2)
    for path_n, path in enumerate(ldf.path):
        l_paths[path_n] = path

    ldf = ldf.iloc[:1]
    ldf['paths'] = ','.join(l_paths)
    ldf['path'] = path_fit_xy
    dfs.append(ldf)

df = pd.concat(dfs)
df = df.drop_duplicates('study_id')
df = df.merge(box_df, on=['study_id'])
dfs = []
for i, idf in df.groupby(['study_id', 'level']):
    l = idf[idf.lr == 'L']
    r = idf[idf.lr == 'R']
    if (len(l) == 0) | (len(r) == 0):
        continue
    idf['l_x'] = (l.x_max.values[0] + l.x_min.values[0])/2
    idf['l_y'] = (l.y_max.values[0] + l.y_min.values[0])/2
    idf['r_x'] = (r.x_max.values[0] + r.x_min.values[0])/2
    idf['r_y'] = (r.y_max.values[0] + r.y_min.values[0])/2
    idf = idf.iloc[:1]
    dfs.append(idf)
df = pd.concat(dfs)    

# tr = pd.read_csv('input/train.csv')
tr = pd.read_csv(f'{WORKING_DIR}/kaggle_csv/train.csv')
df = df.merge(tr, on='study_id')
dfs = []
for level, idf in df.groupby('level'):
    for col in ['spinal_canal_stenosis', 'left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing', 'right_subarticular_stenosis', 'left_subarticular_stenosis']:
        idf[f'{col}_normal'] = 0
        idf[f'{col}_moderate'] = 0
        idf[f'{col}_severe'] = 0
        idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Normal/Mild', f'{col}_normal'] = 1
        idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Moderate', f'{col}_moderate'] = 1
        idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Severe', f'{col}_severe'] = 1
    dfs.append(idf)
df = pd.concat(dfs)
# p = f'input/sagittal_spinal_range2_rolling5.csv'
p = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_spinal_range2_rolling5.csv'
df.to_csv(p, index=False)
print(p)


# foraminal
for left_right in ['left', 'right']:
    dfs = []
    # df_path = 'results/rsna_sagittal_cl/oof.csv'
    df_path = f'{WORKING_DIR}/ckpt/rsna_sagittal_cl/oof.csv'
    df = pd.read_csv(df_path)
    # sdf = pd.read_csv('input/train_series_descriptions.csv')
    sdf = pd.read_csv(f'{WORKING_DIR}/kaggle_csv/train_series_descriptions.csv')
    df = df.merge(sdf, on=['study_id', 'series_id'])
    df = df[df.series_description_y!='Sagittal T1']   
    # df['path'] = f'input/sagittal_all_images/' + df.study_id.astype(str) + '___' + df.instance_number.astype(str) + '.png'
    df['path'] = f'/kaggle/temp/sagittal_all_images/' + df.study_id.astype(str) + '___' + df.instance_number.astype(str) + '.png'
    for id, idf in df.groupby('series_id'):
        idf = idf.sort_values(['x_pos', 'instance_number'])
        idf = idf.drop_duplicates('x_pos')
        idf[f'pred_spinal_rolling'] = idf[f'pred_spinal'].rolling(rolling, center=True).mean()    
        idf[f'pred_{left_right}_neural_rolling'] = idf[f'pred_{left_right}_neural'].rolling(rolling, center=True).mean()
        path_fit_xy = idf[idf['pred_spinal']==idf['pred_spinal'].max()].path.values[0]
        
        col = f'pred_{left_right}_neural_rolling'
        n = idf[idf[col]==idf[col].max()].instance_number.values[0]

        ldf = idf[(idf.instance_number >= n-range_n) & (idf.instance_number <= n+range_n)]
        l_paths = ['nan'] * (1+range_n*2)
        for path_n, path in enumerate(ldf.path):
            l_paths[path_n] = path

        ldf = ldf.iloc[:1]
        ldf['paths'] = ','.join(l_paths)
        ldf['path'] = path_fit_xy
        dfs.append(ldf)
    df = pd.concat(dfs)
    df = df.drop_duplicates('study_id')
    df = df.merge(box_df, on=['study_id'])
    dfs = []
    for i, idf in df.groupby(['study_id', 'level']):
        l = idf[idf.lr == 'L']
        r = idf[idf.lr == 'R']
        if (len(l) == 0) | (len(r) == 0):
            continue
        idf['l_x'] = (l.x_max.values[0] + l.x_min.values[0])/2
        idf['l_y'] = (l.y_max.values[0] + l.y_min.values[0])/2
        idf['r_x'] = (r.x_max.values[0] + r.x_min.values[0])/2
        idf['r_y'] = (r.y_max.values[0] + r.y_min.values[0])/2
        idf = idf.iloc[:1]
        dfs.append(idf)
    df = pd.concat(dfs)    
    # tr = pd.read_csv('input/train.csv')
    tr = pd.read_csv(f'{WORKING_DIR}/kaggle_csv/train.csv')
    df = df.merge(tr, on='study_id')
    dfs = []
    for level, idf in df.groupby('level'):
        for col in ['spinal_canal_stenosis', 'left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing', 'right_subarticular_stenosis', 'left_subarticular_stenosis']:
            idf[f'{col}_normal'] = 0
            idf[f'{col}_moderate'] = 0
            idf[f'{col}_severe'] = 0
            idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Normal/Mild', f'{col}_normal'] = 1
            idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Moderate', f'{col}_moderate'] = 1
            idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Severe', f'{col}_severe'] = 1
        dfs.append(idf)
    df = pd.concat(dfs)    
    # p = f'input/sagittal_{left_right}_nfn_range2_rolling5.csv'
    p = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_{left_right}_nfn_range2_rolling5.csv'
    df.to_csv(p, index=False)
    print(p)


# subarticular
for left_right in ['left', 'right']:
    dfs = []
    # df_path = 'results/rsna_sagittal_cl/oof.csv'
    df_path = f'{WORKING_DIR}/ckpt/rsna_sagittal_cl/oof.csv'
    df = pd.read_csv(df_path)
    # sdf = pd.read_csv('input/train_series_descriptions.csv')
    sdf = pd.read_csv(f'{WORKING_DIR}/kaggle_csv/train_series_descriptions.csv')
    df = df.merge(sdf, on=['study_id', 'series_id'])
    df = df[df.series_description_y!='Sagittal T1']   
    # df['path'] = f'input/sagittal_all_images/' + df.study_id.astype(str) + '___' + df.instance_number.astype(str) + '.png'
    df['path'] = f'/kaggle/temp/sagittal_all_images/' + df.study_id.astype(str) + '___' + df.instance_number.astype(str) + '.png'
    for id, idf in df.groupby('series_id'):
        idf = idf.sort_values(['x_pos', 'instance_number'])
        idf = idf.drop_duplicates('x_pos')
        path_fit_xy = idf[idf['pred_spinal']==idf['pred_spinal'].max()].path.values[0]

        idf[f'pred_spinal_rolling'] = idf[f'pred_spinal'].rolling(rolling, center=True).mean()    
        spinal_n = idf[idf['pred_spinal_rolling']==idf['pred_spinal_rolling'].max()].instance_number.values[0]

        col = f'pred_{left_right}_neural_rolling'
        idf[col] = idf[f'pred_{left_right}_neural'].rolling(rolling, center=True).mean()
        n = idf[idf[col]==idf[col].max()].instance_number.values[0]
        n = (spinal_n + n)//2

        ldf = idf[(idf.instance_number >= n-range_n) & (idf.instance_number <= n+range_n)]
        l_paths = ['nan'] * (1+range_n*2)
        for path_n, path in enumerate(ldf.path):
            l_paths[path_n] = path

        ldf = ldf.iloc[:1]
        ldf['paths'] = ','.join(l_paths)
        ldf['path'] = path_fit_xy
        dfs.append(ldf)
    df = pd.concat(dfs)
    df = df.drop_duplicates('study_id')
    df = df.merge(box_df, on=['study_id'])
    dfs = []
    for i, idf in df.groupby(['study_id', 'level']):
        l = idf[idf.lr == 'L']
        r = idf[idf.lr == 'R']
        if (len(l) == 0) | (len(r) == 0):
            continue
        idf['l_x'] = (l.x_max.values[0] + l.x_min.values[0])/2
        idf['l_y'] = (l.y_max.values[0] + l.y_min.values[0])/2
        idf['r_x'] = (r.x_max.values[0] + r.x_min.values[0])/2
        idf['r_y'] = (r.y_max.values[0] + r.y_min.values[0])/2
        idf = idf.iloc[:1]
        dfs.append(idf)
    df = pd.concat(dfs)    
    # tr = pd.read_csv('input/train.csv')
    tr = pd.read_csv(f'{WORKING_DIR}/kaggle_csv/train.csv')
    df = df.merge(tr, on='study_id')
    dfs = []
    for level, idf in df.groupby('level'):
        for col in ['spinal_canal_stenosis', 'left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing', 'right_subarticular_stenosis', 'left_subarticular_stenosis']:
            idf[f'{col}_normal'] = 0
            idf[f'{col}_moderate'] = 0
            idf[f'{col}_severe'] = 0
            idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Normal/Mild', f'{col}_normal'] = 1
            idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Moderate', f'{col}_moderate'] = 1
            idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Severe', f'{col}_severe'] = 1
        dfs.append(idf)
    df = pd.concat(dfs)    
    # p = f'input/sagittal_{left_right}_ss_range2_rolling5.csv'
    p = f'{WORKING_DIR}/csv_train/axial_classification_7/sagittal_{left_right}_ss_range2_rolling5.csv'
    df.to_csv(p, index=False)
    print(p)
print('preprocess_for_sagittal_classification.py finish')