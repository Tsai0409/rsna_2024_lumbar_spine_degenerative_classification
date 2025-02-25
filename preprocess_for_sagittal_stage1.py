# preprocess_for_sagittal_stage1.py
# sagittal spinal
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings

# kaggle input
DATA_KAGGLE_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"
train_test = "train"

# 設定環境變數
WORKING_DIR="/kaggle/working/duplicate"

warnings.simplefilter('ignore')
pd.set_option('display.max_columns', 100)
print('ready to preprocess_for_sagittal_stage1.py')
#cood = pd.read_csv('input/train_label_coordinates.csv')
cood = pd.read_csv(f'{DATA_KAGGLE_DIR}/train_label_coordinates.csv')
cood['target_level'] = 'none'  # 創立新的 col
cood.loc[(cood.level=='L1/L2') & (cood.condition == 'Spinal Canal Stenosis'), 'target_level'] = 'l1_spinal'  # 當 cood.level 和 cood.condition 條件成立時，target_level = l1_spinal
cood.loc[(cood.level=='L2/L3') & (cood.condition == 'Spinal Canal Stenosis'), 'target_level'] = 'l2_spinal'
cood.loc[(cood.level=='L3/L4') & (cood.condition == 'Spinal Canal Stenosis'), 'target_level'] = 'l3_spinal'
cood.loc[(cood.level=='L4/L5') & (cood.condition == 'Spinal Canal Stenosis'), 'target_level'] = 'l4_spinal'
cood.loc[(cood.level=='L5/S1') & (cood.condition == 'Spinal Canal Stenosis'), 'target_level'] = 'l5_spinal'
cood.loc[(cood.level=='L1/L2') & (cood.condition == 'Left Neural Foraminal Narrowing'), 'target_level'] = 'l1_left_neural'
cood.loc[(cood.level=='L2/L3') & (cood.condition == 'Left Neural Foraminal Narrowing'), 'target_level'] = 'l2_left_neural'
cood.loc[(cood.level=='L3/L4') & (cood.condition == 'Left Neural Foraminal Narrowing'), 'target_level'] = 'l3_left_neural'
cood.loc[(cood.level=='L4/L5') & (cood.condition == 'Left Neural Foraminal Narrowing'), 'target_level'] = 'l4_left_neural'
cood.loc[(cood.level=='L5/S1') & (cood.condition == 'Left Neural Foraminal Narrowing'), 'target_level'] = 'l5_left_neural'
cood.loc[(cood.level=='L1/L2') & (cood.condition == 'Right Neural Foraminal Narrowing'), 'target_level'] = 'l1_right_neural'
cood.loc[(cood.level=='L2/L3') & (cood.condition == 'Right Neural Foraminal Narrowing'), 'target_level'] = 'l2_right_neural'
cood.loc[(cood.level=='L3/L4') & (cood.condition == 'Right Neural Foraminal Narrowing'), 'target_level'] = 'l3_right_neural'
cood.loc[(cood.level=='L4/L5') & (cood.condition == 'Right Neural Foraminal Narrowing'), 'target_level'] = 'l4_right_neural'
cood.loc[(cood.level=='L5/S1') & (cood.condition == 'Right Neural Foraminal Narrowing'), 'target_level'] = 'l5_right_neural'

#train = pd.read_csv('input/train_with_fold.csv')
train = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_4/train_with_fold.csv')
train['instance_number'] = train.path.apply(lambda x: int(x.split('___')[-1].replace('.png', '')))
df = train[train.series_description_x!='Axial T2']  # 留下 Sagittal T1、Sagittal T2/STIR
df[['l1_spinal', 'l2_spinal', 'l3_spinal', 'l4_spinal', 'l5_spinal']] = 0  # 創立新的 col
dfs = []
cs = []  # not use
for id, idf in tqdm(df.groupby('series_id')):  # 對 train_with_fold.csv 做 groupby
    cdf = cood[cood.series_id == id]  # 用 train_with_fold.csv 的 id 找到 train_label_coordinates.csv 對應的 id
    if sorted(cdf.target_level.values) != ['l1_spinal', 'l2_spinal', 'l3_spinal', 'l4_spinal', 'l5_spinal']:
        continue  # 接著檢查 cdf 中所有 target_level 欄位的值是否完整，如果不完整，則用 continue 跳過這個系列，不進行後續標籤更新
    for level in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']:  # 如果 target_level 欄位有完整
        for condition in ['Spinal Canal Stenosis']:
            udf = cdf[(cdf.level== level) & (cdf.condition == condition)]  # 找出 train_label_coordinates.csv 符合條件的；舉例：Spinal Canal Stenosis、L1/L2
            if len(udf)!=0:
                n = udf.instance_number
                idf.loc[idf.instance_number.isin(n), udf.target_level.values[0]] = 1  # idf.instance_number.isin(n) = True -> udf.target_level.values[0] = 1
    dfs.append(idf)

df = pd.concat(dfs)
#df.to_csv('input/train_for_sagittal_level_cl_v1_for_train_spinal_only.csv', index=False)
df.to_csv('train_for_sagittal_level_cl_v1_for_train_spinal_only.csv', index=False)
print('train_for_sagittal_level_cl_v1_for_train_spinal_only.csv finish')


#cood = pd.read_csv('input/train_label_coordinates.csv')
cood = pd.read_csv(f'{DATA_KAGGLE_DIR}/train_label_coordinates.csv')
cood['target_level'] = 'none'
cood.loc[(cood.level=='L1/L2') & (cood.condition == 'Spinal Canal Stenosis'), 'target_level'] = 'l1_spinal'
cood.loc[(cood.level=='L2/L3') & (cood.condition == 'Spinal Canal Stenosis'), 'target_level'] = 'l2_spinal'
cood.loc[(cood.level=='L3/L4') & (cood.condition == 'Spinal Canal Stenosis'), 'target_level'] = 'l3_spinal'
cood.loc[(cood.level=='L4/L5') & (cood.condition == 'Spinal Canal Stenosis'), 'target_level'] = 'l4_spinal'
cood.loc[(cood.level=='L5/S1') & (cood.condition == 'Spinal Canal Stenosis'), 'target_level'] = 'l5_spinal'
cood.loc[(cood.level=='L1/L2') & (cood.condition == 'Left Neural Foraminal Narrowing'), 'target_level'] = 'l1_left_neural'
cood.loc[(cood.level=='L2/L3') & (cood.condition == 'Left Neural Foraminal Narrowing'), 'target_level'] = 'l2_left_neural'
cood.loc[(cood.level=='L3/L4') & (cood.condition == 'Left Neural Foraminal Narrowing'), 'target_level'] = 'l3_left_neural'
cood.loc[(cood.level=='L4/L5') & (cood.condition == 'Left Neural Foraminal Narrowing'), 'target_level'] = 'l4_left_neural'
cood.loc[(cood.level=='L5/S1') & (cood.condition == 'Left Neural Foraminal Narrowing'), 'target_level'] = 'l5_left_neural'
cood.loc[(cood.level=='L1/L2') & (cood.condition == 'Right Neural Foraminal Narrowing'), 'target_level'] = 'l1_right_neural'
cood.loc[(cood.level=='L2/L3') & (cood.condition == 'Right Neural Foraminal Narrowing'), 'target_level'] = 'l2_right_neural'
cood.loc[(cood.level=='L3/L4') & (cood.condition == 'Right Neural Foraminal Narrowing'), 'target_level'] = 'l3_right_neural'
cood.loc[(cood.level=='L4/L5') & (cood.condition == 'Right Neural Foraminal Narrowing'), 'target_level'] = 'l4_right_neural'
cood.loc[(cood.level=='L5/S1') & (cood.condition == 'Right Neural Foraminal Narrowing'), 'target_level'] = 'l5_right_neural'

#train = pd.read_csv('input/train_with_fold.csv')
train = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_4/train_with_fold.csv')
train['instance_number'] = train.path.apply(lambda x: int(x.split('___')[-1].replace('.png', '')))
df = train[train.series_description_x!='Axial T2']
df[['l1_right_neural', 'l2_right_neural', 'l3_right_neural', 'l4_right_neural', 'l5_right_neural', 'l1_left_neural', 'l2_left_neural', 'l3_left_neural', 'l4_left_neural', 'l5_left_neural']] = 0
dfs = []
cs = []  # not use
for id, idf in tqdm(df.groupby('series_id')):
    cdf = cood[cood.series_id == id]
    if ['l1_left_neural', 'l1_right_neural', 'l2_left_neural', 'l2_right_neural', 'l3_left_neural', 'l3_right_neural', 'l4_left_neural', 'l4_right_neural', 'l5_left_neural', 'l5_right_neural'] != sorted(cdf.target_level.values):
        continue  # 直接執行下一次的迭代
    for level in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']:
        for condition in ['Left Neural Foraminal Narrowing', 'Right Neural Foraminal Narrowing']:
            udf = cdf[(cdf.level== level) & (cdf.condition == condition)]
            if len(udf)!=0:
                n = udf.instance_number
                idf.loc[idf.instance_number.isin(n), udf.target_level.values[0]] = 1
    dfs.append(idf)
df = pd.concat(dfs)
#df.to_csv('input/train_for_sagittal_level_cl_v1_for_train_nfn_only.csv', index=False)
df.to_csv('train_for_sagittal_level_cl_v1_for_train_nfn_only.csv', index=False)
print('train_for_sagittal_level_cl_v1_for_train_nfn_only.csv finish')