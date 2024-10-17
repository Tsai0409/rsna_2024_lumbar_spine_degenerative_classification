# sagittal spinal
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')
pd.set_option('display.max_columns', 100)
cood = pd.read_csv('input/train_label_coordinates.csv')
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
train = pd.read_csv('input/train_with_fold.csv')
train['instance_number'] = train.path.apply(lambda x: int(x.split('___')[-1].replace('.png', '')))
df = train[train.series_description!='Axial T2']
df[['l1_spinal', 'l2_spinal', 'l3_spinal', 'l4_spinal', 'l5_spinal']] = 0
dfs = []
cs = []
for id, idf in tqdm(df.groupby('series_id')):
    cdf = cood[cood.series_id == id]
    if sorted(cdf.target_level.values) != ['l1_spinal', 'l2_spinal', 'l3_spinal', 'l4_spinal', 'l5_spinal']:
        continue
    for level in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']:
        for condition in ['Spinal Canal Stenosis']:
            udf = cdf[(cdf.level== level) & (cdf.condition == condition)]
            if len(udf)!=0:
                n = udf.instance_number
                idf.loc[idf.instance_number.isin(n), udf.target_level.values[0]] = 1
    dfs.append(idf)

df = pd.concat(dfs)    
df.to_csv('input/train_for_sagittal_level_cl_v1_for_train_spinal_only.csv', index=False)


cood = pd.read_csv('input/train_label_coordinates.csv')
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
train = pd.read_csv('input/train_with_fold.csv')
train['instance_number'] = train.path.apply(lambda x: int(x.split('___')[-1].replace('.png', '')))
df = train[train.series_description!='Axial T2']
df[['l1_right_neural', 'l2_right_neural', 'l3_right_neural', 'l4_right_neural', 'l5_right_neural', 'l1_left_neural', 'l2_left_neural', 'l3_left_neural', 'l4_left_neural', 'l5_left_neural']] = 0
dfs = []
cs = []
for id, idf in tqdm(df.groupby('series_id')):
    cdf = cood[cood.series_id == id]
    if ['l1_left_neural', 'l1_right_neural', 'l2_left_neural', 'l2_right_neural', 'l3_left_neural', 'l3_right_neural', 'l4_left_neural', 'l4_right_neural', 'l5_left_neural', 'l5_right_neural'] != sorted(cdf.target_level.values):
        continue
    for level in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']:
        for condition in ['Left Neural Foraminal Narrowing', 'Right Neural Foraminal Narrowing']:
            udf = cdf[(cdf.level== level) & (cdf.condition == condition)]
            if len(udf)!=0:
                n = udf.instance_number
                idf.loc[idf.instance_number.isin(n), udf.target_level.values[0]] = 1
    dfs.append(idf)
df = pd.concat(dfs)    
df.to_csv('input/train_for_sagittal_level_cl_v1_for_train_nfn_only.csv', index=False)
