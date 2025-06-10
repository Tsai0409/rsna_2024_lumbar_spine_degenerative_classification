import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics
from pdb import set_trace as st
import copy
import os

import numpy as np
from scipy.special import softmax
import warnings
warnings.simplefilter('ignore')
pd.set_option('display.max_columns', 500)
from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error, average_precision_score, recall_score, log_loss

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).flatten()
    return tn / (tn + fp)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class ParticipantVisibleError(Exception):
    pass

from scipy.optimize import minimize

WORKING_DIR="/kaggle/working/duplicate"

# not used
def get_condition(full_location: str) -> str:
    for injury_condition in ['spinal', 'foraminal', 'subarticular']:
        if injury_condition in full_location:
            return injury_condition
    raise ValueError(f'condition not found in {full_location}')

# sub = pd.read_csv('input/sample_submission.csv')
sub = pd.read_csv(f'{WORKING_DIR}/csv_train/output_2/myself_submission_fold0.csv')

label_features = [
    'spinal_canal_stenosis',
    'left_neural_foraminal_narrowing',
    'right_neural_foraminal_narrowing',
    'left_subarticular_stenosis',
    'right_subarticular_stenosis',
]
true_cols15 = []
true_cols = []
for col in label_features:
    for c in ['normal', 'moderate', 'severe']:
        true_cols15.append(f'{col}_{c}')
for col in label_features:
    for level in ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']:
        for c in ['normal', 'moderate', 'severe']:
            true_cols.append(f'{col}_{level}_{c}')

pred_cols15 = ['pred_'+c for c in true_cols15]  # 15 個 各個病狀的嚴重程度
pred_cols = ['pred_'+c for c in true_cols]  # 75 個 各個病狀的嚴重程度 (包含不同位置)
# tr = pd.read_csv('input/train_with_fold.csv')
tr = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_4/train_with_fold.csv')
t1_ids = tr[tr.series_description_y=='Sagittal T1'].series_id
t2_ids = tr[tr.series_description_y=='Sagittal T2/STIR'].series_id


axial_dis_th = 5


# # axial

configs = [
    'rsna_axial_spinal_dis3_crop_x05_y6',
    'rsna_axial_spinal_dis3_crop_x1_y2',
]
target_pred_cols = [c for c in pred_cols if 'spinal' in c]  # 真實標籤 y_true
target_cols = [c for c in true_cols if 'spinal' in c]       # 預測分數 y_score
config_pred_cols = pred_cols15  # 15 個 各個病狀的嚴重程度

# oof = pd.concat([pd.read_csv(f'results/rsna_axial_spinal_dis3_crop_x1_y2/oof_fold{fold}.csv') for fold in range(5)])  # 合併 5 fold 的結果
oof = pd.concat([pd.read_csv(f'{WORKING_DIR}/ckpt/rsna_axial_spinal_dis3_crop_x1_y2/oof_fold{fold}.csv') for fold in range(1)])
# 確認 column
config_pred_cols = [c for c in config_pred_cols if c in list(oof)]  # 據實際讀入的 oof DataFrame，確認哪些你想要的欄位 config_pred_cols 是真的存在的 -> pred_spinal_canal_stenosis_normal、pred_spinal_canal_stenosis_moderate、pred_spinal_canal_stenosis_severe
config_cols = [col.replace('pred_', '') for col in config_pred_cols]  # -> spinal_canal_stenosis_normal、spinal_canal_stenosis_moderate、spinal_canal_stenosis_severe

oof = oof.groupby(['study_id', 'pred_level'])[config_cols + config_pred_cols].mean().reset_index().sort_values(['study_id', 'pred_level'])  # 將值抓到新的 dataframe
oof.to_csv('oof.csv')  # 我加
true = oof[config_cols].values  # -> spinal_canal_stenosis_normal、spinal_canal_stenosis_moderate、spinal_canal_stenosis_severe (主要找到 true label)

dfs = []
for config in configs:
    # oof = pd.concat([pd.read_csv(f'results/{config}/oof_fold{fold}.csv') for fold in range(5)])
    oof = pd.concat([pd.read_csv(f'{WORKING_DIR}/ckpt/{config}/oof_fold{fold}.csv') for fold in range(1)])
    # score = np.mean([np.mean([roc_auc_score(oof[oof.pred_level==l][col.replace('pred_', '')], oof[oof.pred_level==l][col]) for col in config_pred_cols]) for l in [1,2,3,4,5]])
    # score2 = np.mean([np.mean([log_loss(oof[oof.pred_level==l][col.replace('pred_', '')], sigmoid(oof[oof.pred_level==l][col])) for col in config_pred_cols]) for l in [1,2,3,4,5]])
    score = np.mean([  # 整個 dataframe 算一個
        np.mean([
            roc_auc_score(  #  ROC AUC 分數(Area Under Curve)，用來評估預測效果好壞
                oof[oof.pred_level==l][col.replace('pred_', '')],  # 真實標籤 y_true -> y_true → 實際是否為 normal
                oof[oof.pred_level==l][col]                        # 預測分數 y_score -> y_score → 模型預測值
            ) for col in config_pred_cols  #  -> pred_spinal_canal_stenosis_normal、pred_spinal_canal_stenosis_moderate、pred_spinal_canal_stenosis_severe
        ]) for l in [1,2,3,4,5]
    ])
    score2 = np.mean([  # 整個 dataframe 算一個
        np.mean([
            log_loss(
                oof[oof.pred_level==l][col.replace('pred_', '')], 
                sigmoid(oof[oof.pred_level==l][col])
            ) for col in config_pred_cols
        ]) for l in [1,2,3,4,5]
    ])
    
    print(len(oof), round(score, 4), round(score2, 4), config)
    oof = oof.groupby(['study_id', 'pred_level'])[config_pred_cols].mean().reset_index().sort_values(['study_id', 'pred_level'])  # 針對 5 個 fold 做 mean()
    dfs.append(oof)  # 將每個 group 合併
oof = pd.concat(dfs)  # 將每個 configs 合併

oof = oof.groupby(['study_id', 'pred_level'])[config_pred_cols].mean().reset_index()  # 對不同 configs 做 mean()
oof[config_cols] = true  # -> spinal_canal_stenosis_normal、spinal_canal_stenosis_moderate、spinal_canal_stenosis_severe

oof[[col.replace('pred_', '') for col in config_pred_cols]] = oof[[col.replace('pred_', '') for col in config_pred_cols]].astype(int)  # 將 spinal_canal_stenosis_normal、spinal_canal_stenosis_moderate、spinal_canal_stenosis_severe 轉為整數
oof[['normal', 'moderate', 'severe']] = oof[[c.replace('pred_', '') for c in config_pred_cols]].values  # 將 spinal_canal_stenosis_normal、spinal_canal_stenosis_moderate、spinal_canal_stenosis_severe -> normal、moderate、severe(標準答案)
oof[['pred_normal', 'pred_moderate', 'pred_severe']] = oof[config_pred_cols].values  # 將 pred_spinal_canal_stenosis_normal、pred_spinal_canal_stenosis_moderate、pred_spinal_canal_stenosis_severe -> pred_normal、pred_moderate、pred_severe(預測答案)
oof.to_csv('oof2.csv')  # 我加
axial_spinal = oof.copy()


# axial nfn

configs = [
    'rsna_axial_ss_nfn_x2_y2_center_pad0',
    'rsna_axial_ss_nfn_x2_y6_center_pad0',
    'rsna_axial_ss_nfn_x2_y8_center_pad10',
]
target_pred_cols = [c for c in pred_cols if 'neural_foraminal_narrowing' in c]  # 75 個 各個病狀的嚴重程度 (nfn(left/right) L1~L5 n/m/s)
target_cols = [c for c in true_cols if 'neural_foraminal_narrowing' in c]  # 75 個 各個病狀的嚴重程度 (nfn(left/right) L1~L5 n/m/s)
cols = [
    'neural_foraminal_narrowing_normal',
    'neural_foraminal_narrowing_moderate',
    'neural_foraminal_narrowing_severe'
]
config_pred_cols = ['pred_'+c for c in cols]
config_cols = [col.replace('pred_', '') for col in config_pred_cols]

preds = []
for config in configs:
    # oof = pd.concat([pd.read_csv(f'results/{config}/oof_fold{fold}.csv') for fold in range(5)])
    oof = pd.concat([pd.read_csv(f'{WORKING_DIR}/ckpt/{config}/oof_fold{fold}.csv') for fold in range(1)])
    score = np.mean([log_loss(oof[col.replace('pred_', '')], sigmoid(oof[col])) for col in config_pred_cols])
    score2 = np.mean([roc_auc_score(oof[col.replace('pred_', '')], sigmoid(oof[col])) for col in config_pred_cols])
    print(len(oof), round(score, 4), round(score2, 4), config)
    preds.append(oof[config_pred_cols].values)  # 將值存成 list 的形式 -> neural_foraminal_narrowing_normal、neural_foraminal_narrowing_moderate、neural_foraminal_narrowing_severe
oof[config_pred_cols] = np.mean(preds, 0)  # 以每個 config 取 mean()
oof[['normal', 'moderate', 'severe']] = oof[[c.replace('pred_', '') for c in config_pred_cols]].values
oof[['pred_normal', 'pred_moderate', 'pred_severe']] = oof[config_pred_cols].values
oof.loc[oof.dis.isnull(), 'dis'] = oof.dis.mean()
oof.to_csv('oof3.csv')  # 我加
axial_nfn = oof[oof.dis < axial_dis_th]  # axial_dis_th = 5
axial_nfn.to_csv('axial_nfn.csv')  # 我加


# axial ss
configs = [
    'rsna_axial_ss_nfn_x2_y2_center_pad0',
    'rsna_axial_ss_nfn_x2_y6_center_pad0',
    'rsna_axial_ss_nfn_x2_y8_center_pad10',
]
cols = [
    'subarticular_stenosis_normal', 
    'subarticular_stenosis_moderate', 
    'subarticular_stenosis_severe'
]
# oof = pd.concat([pd.read_csv(f'results/{configs[0]}/oof_fold{fold}.csv') for fold in range(5)])
oof = pd.concat([pd.read_csv(f'{WORKING_DIR}/ckpt/{configs[0]}/oof_fold{fold}.csv') for fold in range(1)])
config_pred_cols = ['pred_'+c for c in cols]
config_cols = [col.replace('pred_', '') for col in config_pred_cols]

preds = []
for config in configs:
    # oof = pd.concat([pd.read_csv(f'results/{config}/oof_fold{fold}.csv') for fold in range(5)])  #.sort_values(['path', 'level'])、
    oof = pd.concat([pd.read_csv(f'{WORKING_DIR}/ckpt/{config}/oof_fold{fold}.csv') for fold in range(1)])
    score = np.mean([log_loss(oof[col.replace('pred_', '')], sigmoid(oof[col])) for col in config_pred_cols])
    score2 = np.mean([roc_auc_score(oof[col.replace('pred_', '')], sigmoid(oof[col])) for col in config_pred_cols])
    oof = oof[oof.dis < axial_dis_th]
    print(len(oof), round(score, 4), round(score2, 4), config)
    
    preds.append(oof[config_pred_cols].values)
oof[config_pred_cols] = np.mean(preds, 0)
oof[['normal', 'moderate', 'severe']] = oof[[c.replace('pred_', '') for c in config_pred_cols]].values
oof[['pred_normal', 'pred_moderate', 'pred_severe']] = oof[config_pred_cols].values
oof.loc[oof.dis.isnull(), 'dis'] = oof.dis.mean()
axial_ss = oof[oof.dis < axial_dis_th]  # axial_dis_th = 5
axial_ss.to_csv('axial_ss.csv')  # 我加


# # sagittal

# spinal
configs = [
    'rsna_saggital_mil_spinal_crop_x03_y05',
    'rsna_saggital_mil_spinal_crop_x03_y07',     
]
config_cols = [
    'spinal_canal_stenosis_normal',
    'spinal_canal_stenosis_moderate',
    'spinal_canal_stenosis_severe',
]

config_pred_cols = ['pred_'+c for c in config_cols]
preds = []
for config in configs:
    # oof = pd.concat([pd.read_csv(f'results/{config}/oof_fold{fold}.csv') for fold in range(5)])
    oof = pd.concat([pd.read_csv(f'{WORKING_DIR}/ckpt/{config}/oof_fold{fold}.csv') for fold in range(1)])
    oof['pred_level'] = oof.level.map({
        'L1/L2': 1,
        'L2/L3': 2,
        'L3/L4': 3,
        'L4/L5': 4,
        'L5/S1': 5,
    })    
        
    # score = np.mean([np.mean([log_loss(oof[oof.level==l][col.replace('pred_', '')], sigmoid(oof[oof.level==l][col])) for col in config_pred_cols]) for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']])
    score = np.mean([
        np.mean([
            log_loss(
                oof[oof.level == l][col.replace('pred_', '')], 
                sigmoid(oof[oof.level == l][col]),
                labels=[0, 1] # 我加
            ) 
            for col in config_pred_cols
            if len(np.unique(oof[oof.level == l][col.replace('pred_', '')])) > 1  # 我加
        ]) 
        for l in ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
    ])
    
    print(len(oof), round(score, 4), config)
    preds.append(oof[config_pred_cols].values)
oof[config_pred_cols] = np.mean(preds, 0)  # 以每個 config 取 mean()
oof[['normal', 'moderate', 'severe']] = oof[[c.replace('pred_', '') for c in config_pred_cols]].values
oof[['pred_normal', 'pred_moderate', 'pred_severe']] = oof[config_pred_cols].values
sagittal_spinal = oof.copy()
sagittal_spinal.to_csv('sagittal_spinal.csv')  # 我加


# nfn
configs = [
    'rsna_saggital_mil_nfn_crop_x07_y1_v2',
    'rsna_saggital_mil_nfn_crop_x15_y1_v2',
    'rsna_saggital_mil_nfn_crop_x03_y1_v2',
]

config_cols = ['neural_foraminal_narrowing_normal', 'neural_foraminal_narrowing_moderate', 'neural_foraminal_narrowing_severe']
config_pred_cols = ['pred_'+c for c in config_cols]

preds = []
for config in configs:
    # oof = pd.concat([pd.read_csv(f'results/{config}/oof_fold{fold}.csv') for fold in range(5)])
    oof = pd.concat([pd.read_csv(f'{WORKING_DIR}/ckpt/{config}/oof_fold{fold}.csv') for fold in range(1)])
    score = np.mean([log_loss(oof[col.replace('pred_', '')], sigmoid(oof[col])) for col in config_pred_cols])
    score2 = np.mean([roc_auc_score(oof[col.replace('pred_', '')], sigmoid(oof[col])) for col in config_pred_cols])
    print(len(oof), round(score, 4), round(score2, 4), config)
    
    oof['pred_level'] = oof.level.map({
        'L1/L2': 1,
        'L2/L3': 2,
        'L3/L4': 3,
        'L4/L5': 4,
        'L5/S1': 5,
    })    
    if 'left_nfn' in list(oof):
        oof['left_right'] = 'right'
        oof.loc[oof.left_nfn == 1, 'left_right'] = 'left'
    preds.append(oof[config_pred_cols].values)
oof[config_pred_cols] = np.mean(preds, 0)
oof[['pred_normal', 'pred_moderate', 'pred_severe']] = oof[config_pred_cols].values
oof[['normal', 'moderate', 'severe']] = oof[[c.replace('pred_', '') for c in config_pred_cols]].values
sagittal_nfn = oof.copy()


# ss
configs=[
    'rsna_saggital_mil_ss_crop_x03_y05_96',
    'rsna_saggital_mil_ss_crop_x03_y07_96',
    'rsna_saggital_mil_ss_crop_x03_y2_96',
    'rsna_saggital_mil_ss_crop_x1_y07_96',
]
config_cols = ['subarticular_stenosis_normal', 'subarticular_stenosis_moderate', 'subarticular_stenosis_severe']
config_pred_cols = ['pred_'+c for c in config_cols]
preds = []
for config in configs:
    # oof = pd.concat([pd.read_csv(f'results/{config}/oof_fold{fold}.csv') for fold in range(5)])
    oof = pd.concat([pd.read_csv(f'{WORKING_DIR}/ckpt/{config}/oof_fold{fold}.csv') for fold in range(1)])
    score = np.mean([log_loss(oof[col.replace('pred_', '')], sigmoid(oof[col])) for col in config_pred_cols])
    score2 = np.mean([roc_auc_score(oof[col.replace('pred_', '')], sigmoid(oof[col])) for col in config_pred_cols])
    print(len(oof), round(score, 4), round(score2, 4), config)
    
    if 'left_nfn' in list(oof):
        oof['left_right'] = 'right'
        oof.loc[oof.left_nfn == 1, 'left_right'] = 'left'
    oof = oof.groupby(['study_id', 'level', 'left_right'])[config_cols+config_pred_cols].mean().reset_index()
    oof['pred_level'] = oof.level.map({
        'L1/L2': 1,
        'L2/L3': 2,
        'L3/L4': 3,
        'L4/L5': 4,
        'L5/S1': 5,
    })    
    preds.append(oof[config_pred_cols].values)
oof[config_pred_cols] = np.mean(preds, 0)
oof[['normal', 'moderate', 'severe']] = oof[[c.replace('pred_', '') for c in config_pred_cols]].values
oof[['pred_normal', 'pred_moderate', 'pred_severe']] = oof[config_pred_cols].values
sagittal_ss = oof.copy()


study_ids = []
targets = []
levels = []
is_axials = []
level_preds = []
lrs = []
trues = []
preds = []


for (study_id, level), idf in axial_spinal.groupby(['study_id', 'pred_level']):
    trues.append(idf[['normal', 'moderate', 'severe']].mean(0).values)
    preds.append(idf[['pred_normal', 'pred_moderate', 'pred_severe']].mean(0).values)
    study_ids.append(study_id)
    targets.append('spinal')
    levels.append(level)
    is_axials.append(1)
    lrs.append('center')


for (study_id, level, lr), idf in axial_nfn.groupby(['study_id', 'pred_level', 'left_right']):
    trues.append(idf[['normal', 'moderate', 'severe']].mean(0).values)
    preds.append(idf[['pred_normal', 'pred_moderate', 'pred_severe']].mean(0).values)
    study_ids.append(study_id)
    targets.append('nfn')
    levels.append(level)
    is_axials.append(1)
    lrs.append(lr)


for (study_id, level, lr), idf in axial_ss.groupby(['study_id', 'pred_level', 'left_right']):
    trues.append(idf[['normal', 'moderate', 'severe']].mean(0).values)
    preds.append(idf[['pred_normal', 'pred_moderate', 'pred_severe']].mean(0).values)
    study_ids.append(study_id)
    targets.append('ss')
    levels.append(level)
    is_axials.append(1)
    lrs.append(lr)


for (study_id, level), idf in sagittal_spinal.groupby(['study_id', 'pred_level']):
    trues.append(idf[['normal', 'moderate', 'severe']].mean(0).values)
    preds.append(idf[['pred_normal', 'pred_moderate', 'pred_severe']].mean(0).values)
    study_ids.append(study_id)
    targets.append('spinal')
    levels.append(level)
    is_axials.append(0)
    lrs.append('center')


for (study_id, level, lr), idf in sagittal_nfn.groupby(['study_id', 'pred_level', 'left_right']):
    trues.append(idf[['normal', 'moderate', 'severe']].mean(0).values)
    preds.append(idf[['pred_normal', 'pred_moderate', 'pred_severe']].mean(0).values)
    study_ids.append(study_id)
    targets.append('nfn')
    levels.append(level)
    is_axials.append(0)
    lrs.append(lr)


for (study_id, level, lr), idf in sagittal_ss.groupby(['study_id', 'pred_level', 'left_right']):
    trues.append(idf[['normal', 'moderate', 'severe']].mean(0).values)
    preds.append(idf[['pred_normal', 'pred_moderate', 'pred_severe']].mean(0).values)
    study_ids.append(study_id)
    targets.append('ss')
    levels.append(level)
    is_axials.append(0)
    lrs.append(lr)


oof = pd.DataFrame({
    'study_id': study_ids,
    'target': targets,
    'level': levels,
    'is_axial': is_axials,
    'lr': lrs,
})
oof[['normal', 'moderate', 'severe']] = np.array(trues).astype(int)
oof[['pred_normal', 'pred_moderate', 'pred_severe']] = np.array(preds)
oof.to_csv('oof4.csv')  # 我加


axial = oof[oof.is_axial==1]
axial.head(2)
axial.columns = [
    'study_id',
    'target',
    'level',
    'is_axial',
    'lr',
    'normal',
    'moderate',
    'severe',
    'axial_pred_normal',  # 將 ['pred_normal', 'pred_moderate', 'pred_severe'] -> ['axial_pred_normal', 'axial_pred_moderate', 'axial_pred_severe'] 重新命名
    'axial_pred_moderate',
    'axial_pred_severe'
]
del axial['is_axial']

sagittal = oof[oof.is_axial==0]
sagittal.columns = [
    'study_id',
    'target',
    'level',
    'is_axial',
    'lr',
    'normal',
    'moderate',
    'severe',
    'sagittal_pred_normal',
    'sagittal_pred_moderate',
    'sagittal_pred_severe'
]

for c in ['is_axial', 'normal', 'moderate', 'severe']:
    del sagittal[c]


df = axial.merge(sagittal, on=['study_id', 'target', 'level', 'lr'], how='outer')  # how='outer'：保留 兩邊的所有資料

df.loc[df.sagittal_pred_normal.isnull(), 'sagittal_pred_normal'] = df.loc[df.sagittal_pred_normal.isnull(), 'axial_pred_normal']
df.loc[df.sagittal_pred_moderate.isnull(), 'sagittal_pred_moderate'] = df.loc[df.sagittal_pred_moderate.isnull(), 'axial_pred_moderate']
df.loc[df.sagittal_pred_severe.isnull(), 'sagittal_pred_severe'] = df.loc[df.sagittal_pred_severe.isnull(), 'axial_pred_severe']

df.loc[df.axial_pred_normal.isnull(), 'axial_pred_normal'] = df.loc[df.axial_pred_normal.isnull(), 'sagittal_pred_normal']
df.loc[df.axial_pred_moderate.isnull(), 'axial_pred_moderate'] = df.loc[df.axial_pred_moderate.isnull(), 'sagittal_pred_moderate']
df.loc[df.axial_pred_severe.isnull(), 'axial_pred_severe'] = df.loc[df.axial_pred_severe.isnull(), 'sagittal_pred_severe']

df.loc[((df.target=='nfn') & (df.lr == 'left')), 'target'] = 'left_neural_foraminal_narrowing'
df.loc[((df.target=='nfn') & (df.lr == 'right')), 'target'] = 'right_neural_foraminal_narrowing'
df.loc[((df.target=='ss') & (df.lr == 'left')), 'target'] = 'left_subarticular_stenosis'
df.loc[((df.target=='ss') & (df.lr == 'right')), 'target'] = 'right_subarticular_stenosis'
df.loc[(df.target=='spinal'), 'target'] = 'spinal_canal_stenosis'
df.level = df.level.map({
    1: 'l1_l2',
    2: 'l2_l3',
    3: 'l3_l4',
    4: 'l4_l5',
    5: 'l5_s1',
})
df=df.sort_values(['study_id', 'target'])
df.to_csv('df.csv')  # 我加

m = {
    'study_id': []
}

meta_cols = []
for axial_sagittal in ['sagittal', 'axial']:
    for target in ['spinal_canal_stenosis', 'left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing', 'left_subarticular_stenosis', 'right_subarticular_stenosis']:
        for level in ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']:
            for condition in ['normal', 'moderate', 'severe']:
                m[f'{axial_sagittal}_pred_{target}_{level}_{condition}'] = []
                meta_cols.append(f'{axial_sagittal}_pred_{target}_{level}_{condition}')

ts  = []
for i, idf in df.groupby('study_id'):  # 拆成 152 個 col (將 axial、sagittal 分開)
    m['study_id'].append(i)
    for level in ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']:
        ldf = idf[idf.level == level]
        for target in ['spinal_canal_stenosis', 'left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing', 'left_subarticular_stenosis', 'right_subarticular_stenosis']:
            tdf = ldf[ldf.target == target]
            for condition in ['normal', 'moderate', 'severe']:
                if len(tdf) == 0:
                    pass
                else:
                    assert len(tdf) == 1
                
                for axial_sagittal in ['sagittal', 'axial']:
                    if len(tdf) == 0:
                        m[f'{axial_sagittal}_pred_{target}_{level}_{condition}'].append(0)
                    else:
                        assert len(tdf) == 1
                        m[f'{axial_sagittal}_pred_{target}_{level}_{condition}'].append(tdf[f'{axial_sagittal}_pred_{condition}'].values[0])
df = pd.DataFrame(m)
df.to_csv('df2.csv')  # 我加


# tr = pd.read_csv('input/train.csv')
tr = pd.read_csv(f'{WORKING_DIR}/kaggle_csv/train.csv')
# label_features 有 25 個狀態
label_features = ['spinal_canal_stenosis_l1_l2', 'spinal_canal_stenosis_l2_l3', 'spinal_canal_stenosis_l3_l4', 'spinal_canal_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1', 'left_neural_foraminal_narrowing_l1_l2', 'left_neural_foraminal_narrowing_l2_l3', 'left_neural_foraminal_narrowing_l3_l4', 'left_neural_foraminal_narrowing_l4_l5', 'left_neural_foraminal_narrowing_l5_s1', 'right_neural_foraminal_narrowing_l1_l2', 'right_neural_foraminal_narrowing_l2_l3', 'right_neural_foraminal_narrowing_l3_l4', 'right_neural_foraminal_narrowing_l4_l5', 'right_neural_foraminal_narrowing_l5_s1', 'left_subarticular_stenosis_l1_l2', 'left_subarticular_stenosis_l2_l3', 'left_subarticular_stenosis_l3_l4', 'left_subarticular_stenosis_l4_l5', 'left_subarticular_stenosis_l5_s1', 'right_subarticular_stenosis_l1_l2', 'right_subarticular_stenosis_l2_l3', 'right_subarticular_stenosis_l3_l4', 'right_subarticular_stenosis_l4_l5', 'right_subarticular_stenosis_l5_s1']
cols = []  # 75 個狀態
for col in label_features:
    tr[f'{col}_normal'] = 0
    tr[f'{col}_moderate'] = 0
    tr[f'{col}_severe'] = 0
    tr.loc[tr[col]=='Normal/Mild', f'{col}_normal'] = 1  # 將 train.csv 的 25 個狀態 -> 轉為 75 個狀態的 one-hot vector(_normal、_moderate、_severe)
    tr.loc[tr[col]=='Moderate', f'{col}_moderate'] = 1
    tr.loc[tr[col]=='Severe', f'{col}_severe'] = 1
    tr.loc[tr[col].isnull(), f'{col}_normal'] = np.nan
    tr.loc[tr[col].isnull(), f'{col}_moderate'] = np.nan
    tr.loc[tr[col].isnull(), f'{col}_severe'] = np.nan
    cols.append(f'{col}_normal')
    cols.append(f'{col}_moderate')
    cols.append(f'{col}_severe')

tr = tr[['study_id']+cols]
for c in cols:
    if c in list(df):
        del df[c]
oof = tr.merge(df, on='study_id')
for c in cols:
    oof.loc[oof[c].isnull(), 'axial_pred_'+c] = np.nan
    oof.loc[oof[c].isnull(), 'sagittal_pred_'+c] = np.nan
oof.to_csv('oof5.csv')  # 我加 -> 75 個實際 label + 150 預測 label(axial、sagittal)


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# 載入你的 oof 結果（如果已經載入就不用這行）
oof = pd.read_csv('oof5.csv')  # 這裡路徑依你自己的為主

# 定義 25 個條件（label_features）
label_features = [
    'spinal_canal_stenosis_l1_l2', 'spinal_canal_stenosis_l2_l3', 'spinal_canal_stenosis_l3_l4', 'spinal_canal_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1',
    'left_neural_foraminal_narrowing_l1_l2', 'left_neural_foraminal_narrowing_l2_l3', 'left_neural_foraminal_narrowing_l3_l4', 'left_neural_foraminal_narrowing_l4_l5', 'left_neural_foraminal_narrowing_l5_s1',
    'right_neural_foraminal_narrowing_l1_l2', 'right_neural_foraminal_narrowing_l2_l3', 'right_neural_foraminal_narrowing_l3_l4', 'right_neural_foraminal_narrowing_l4_l5', 'right_neural_foraminal_narrowing_l5_s1',
    'left_subarticular_stenosis_l1_l2', 'left_subarticular_stenosis_l2_l3', 'left_subarticular_stenosis_l3_l4', 'left_subarticular_stenosis_l4_l5', 'left_subarticular_stenosis_l5_s1',
    'right_subarticular_stenosis_l1_l2', 'right_subarticular_stenosis_l2_l3', 'right_subarticular_stenosis_l3_l4', 'right_subarticular_stenosis_l4_l5', 'right_subarticular_stenosis_l5_s1'
]

true_labels = []
pred_labels = []

for feat in label_features:
    # 取 true one-hot
    y_true_onehot = oof[[f'{feat}_normal', f'{feat}_moderate', f'{feat}_severe']].values
    # 取 pred 機率
    y_pred_proba = oof[[f'pred_{feat}_normal', f'pred_{feat}_moderate', f'pred_{feat}_severe']].values

    # nan 跳過（未標註的資料）
    mask = ~np.isnan(y_true_onehot).any(axis=1)
    y_true_onehot = y_true_onehot[mask]
    y_pred_proba = y_pred_proba[mask]
    if len(y_true_onehot) == 0:
        continue

    # one-hot 轉 index (0: normal, 1: moderate, 2: severe)
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    true_labels.append(y_true)
    pred_labels.append(y_pred)


from sklearn.metrics import confusion_matrix, classification_report

# 全部合併成 1D
true_labels = np.concatenate(true_labels)
pred_labels = np.concatenate(pred_labels)

# confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])
print('Confusion Matrix (全部 label_features 攤平成一份):')
print(cm)
print('\n[0] normal, [1] moderate, [2] severe\n')

# 詳細分類報告
print(classification_report(true_labels, pred_labels, target_names=['normal', 'moderate', 'severe']))

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['normal', 'moderate', 'severe'],
            yticklabels=['normal', 'moderate', 'severe'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap (All labels)")
plt.tight_layout()
plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F

# ① 自定義 Loss 函數與工具函數定義
def generate_weights_from_onehot(targets, class_weights):
    y_true_indices = torch.argmax(targets, dim=1)  # targets = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] -> y_true_indices = [0, 1, 2]
    weights = class_weights[y_true_indices]
    return weights

class Rsna2024Loss(nn.Module):
    def __init__(self):
        super(Rsna2024Loss, self).__init__()
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.epsilon = 1e-7

    def forward(self, preds, targets, sigmoid=True):
        if sigmoid:  # sigmoid = False
            preds = torch.sigmoid(preds)
        loss_list = []
        for idxes, class_weights in zip([range(15),range(15, 45),range(45, 75)], [[1,2,4]*5,[1,2,4]*10, [1,2,4]*10]):
            group_targets = targets[:, idxes]  # (batch size, selected col)
            group_targets = group_targets.view(group_targets.shape[0]*len(idxes)//3, 3)  # batch size * selected col / 3(normal, moderate, severe)
            group_preds = preds[:, idxes]
            group_preds = group_preds.view(group_preds.shape[0]*len(idxes)//3, 3)
            weights = generate_weights_from_onehot(group_targets, torch.tensor(class_weights, device=self.device))
            weights = weights * group_targets.sum(1)
            group_preds = group_preds / torch.sum(group_preds, dim=1, keepdim=True)
            loss = - (group_targets * group_preds.log()).sum(dim=1)  # 計算每一筆樣本的 multi-class cross-entropy loss
            loss = (weights * loss).sum() / weights.sum()
            loss_list.append(loss)

        idxes = range(15)  # 針對 spinal
        group_targets = targets[:, idxes]
        group_preds = preds[:, idxes]
        group_targets = torch.max(group_targets[:, [2,5,8,11,14]], 1)[0]  # 針對 severe
        group_preds = torch.max(group_preds[:, [2,5,8,11,14]], 1)[0]
        weights = (targets[:, idxes]*torch.tensor([[1,2,4]*5]*len(targets), device=self.device)).max(1)[0]
        losses = F.binary_cross_entropy(group_preds, group_targets, reduction='none')
        loss = torch.sum(losses * weights) / torch.sum(weights)
        loss_list.append(loss)
        avg_loss = (loss_list[0] + loss_list[1] + loss_list[2] + loss_list[3]) / 4
        return avg_loss, loss_list

cri = Rsna2024Loss()

def normalize_probabilities_to_one_torch(tensor: torch.Tensor) -> torch.Tensor:
    # Normalize the sum of each row's probabilities to 100%.
    # 0.75, 0.75 => 0.5, 0.5
    # 0.1, 0.1 => 0.5, 0.5
    row_totals = tensor.sum(dim=1, keepdim=True)
    if (row_totals == 0).any():
        return tensor
        raise ValueError('All rows must contain at least one non-zero prediction')
    normalized_tensor = tensor / row_totals
    return normalized_tensor


import torch

# not used
def custom_normalize_torch(tensor: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensor has the correct shape (batch_size, 3)
    if tensor.shape[1] != 3:
        raise ValueError("Input tensor must have shape (batch_size, 3)")

    # Split the tensor into normal, moderate, and severe probabilities
    normal, moderate, severe = tensor.unbind(dim=1)

    # Create a mask for rows where severe is the highest probability
    severe_highest = (severe >= torch.max(normal, moderate))

    # Calculate the remaining probability for severe highest cases
    remaining_prob = torch.clamp(1 - severe, min=0)

    # Normalize the other two probabilities for severe highest cases
    total = normal + moderate
    total = torch.where(total == 0, torch.ones_like(total), total)  # Avoid division by zero
    normal_normalized = (normal / total) * remaining_prob
    moderate_normalized = (moderate / total) * remaining_prob

    # Regular normalization for non-severe highest cases
    row_totals = tensor.sum(dim=1, keepdim=True)
    row_totals = torch.where(row_totals == 0, torch.ones_like(row_totals), row_totals)  # Avoid division by zero
    regular_normalized = tensor / row_totals

    # Combine the results using the mask
    result = torch.where(
        severe_highest.unsqueeze(1),
        torch.stack([normal_normalized, moderate_normalized, severe], dim=1),
        regular_normalized
    )
    return result

# ② axial + sagittal 融合預測 + normalize + severe 放大加權
ws = [0.7, 0.5, 0.8]
for condition, w in zip(['spinal', 'neural_foraminal_narrowing', 'subarticular_stenosis'], ws):    
    c_cols = [c for c in pred_cols if condition in c]
    for c in c_cols:
        oof[c] = oof['axial_'+c]*w + oof['sagittal_'+c]*(1-w)
        oof[c] = oof[c].fillna(oof[c].mean())  # 補齊缺失值
preds = sigmoid(oof[pred_cols].fillna(0).values)
preds = torch.FloatTensor(preds)
for i in range(5):  # spinal
    preds[:, i*3+1] *= 1.8  # moderate (1,4,7,10,13)
    preds[:, i*3+2] *= 5    # severe (2,5,8,11,14) -> (大範圍加強)
    preds[:, i*3:(i+1)*3] = normalize_probabilities_to_one_torch(preds[:, i*3:(i+1)*3])  # 重新正規化，針對 0-14
for i in range(5, 15):  # neural_foraminal_narrowing
    preds[:, i*3+1] *= 2.2
    preds[:, i*3+2] *= 5
    preds[:, i*3:(i+1)*3] = normalize_probabilities_to_one_torch(preds[:, i*3:(i+1)*3])
for i in range(15, 25):  # subarticular_stenosis
    preds[:, i*3+1] *= 2.2
    preds[:, i*3+2] *= 5.5
    preds[:, i*3:(i+1)*3] = normalize_probabilities_to_one_torch(preds[:, i*3:(i+1)*3])


# ③ 呼叫 Loss 函數進行評估
import copy
preds_yuji = copy.deepcopy(preds)
oof[pred_cols] = preds_yuji.numpy()
trues = torch.FloatTensor(oof[[c.replace('pred_', '') for c in pred_cols]].fillna(0).values.astype(int))
len(oof), cri(preds_yuji, trues, False)

# not used
def normalize_probabilities_to_one_numpy(array: np.ndarray) -> np.ndarray:
    total = np.sum(array)
    if total == 0:
        return array
    normalized_array = array / total
    return normalized_array


# df = pd.read_csv('input/oof_predictions_submission_format_ip_v4_sub.csv')
# df['study_id'] = df.row_id.apply(lambda x: int(x.split('_')[0]))
# print(len(df))
# df['target_level'] = df.row_id.apply(lambda x: '_'.join(x.split('_')[1:]))
# df['target'] = df.row_id.apply(lambda x: '_'.join(x.split('_')[1:-2]))
# df['level'] = df.row_id.apply(lambda x: '_'.join(x.split('_')[-2:]))
# cols = ['spinal_canal_stenosis_l1_l2', 'spinal_canal_stenosis_l2_l3', 'spinal_canal_stenosis_l3_l4', 'spinal_canal_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1', 'left_neural_foraminal_narrowing_l1_l2', 'left_neural_foraminal_narrowing_l2_l3', 'left_neural_foraminal_narrowing_l3_l4', 'left_neural_foraminal_narrowing_l4_l5', 'left_neural_foraminal_narrowing_l5_s1', 'right_neural_foraminal_narrowing_l1_l2', 'right_neural_foraminal_narrowing_l2_l3', 'right_neural_foraminal_narrowing_l3_l4', 'right_neural_foraminal_narrowing_l4_l5', 'right_neural_foraminal_narrowing_l5_s1', 'left_subarticular_stenosis_l1_l2', 'left_subarticular_stenosis_l2_l3', 'left_subarticular_stenosis_l3_l4', 'left_subarticular_stenosis_l4_l5', 'left_subarticular_stenosis_l5_s1', 'right_subarticular_stenosis_l1_l2', 'right_subarticular_stenosis_l2_l3', 'right_subarticular_stenosis_l3_l4', 'right_subarticular_stenosis_l4_l5', 'right_subarticular_stenosis_l5_s1']

# m = {'study_id': []}
# for c in cols:
#     m[f'ian_{c}_normal'] = []
#     m[f'ian_{c}_moderate'] = []
#     m[f'ian_{c}_severe'] = []
# for i, idf in df.groupby('study_id'):
#     m['study_id'].append(i)

#     for tl, ldf in idf.groupby('target_level'):
#         m[f'ian_{tl}_normal'].append(ldf.normal_mild.values[0])
#         m[f'ian_{tl}_moderate'].append(ldf.moderate.values[0])
#         m[f'ian_{tl}_severe'].append(ldf.severe.values[0])
# ian = pd.DataFrame(m).sort_values('study_id')
# preds_ian = torch.FloatTensor(ian[[c.replace('pred_', 'ian_') for c in pred_cols]].fillna(0).values)
# trues = torch.FloatTensor(oof[[c.replace('pred_', '') for c in pred_cols]].fillna(0).values.astype(int))
# cri(preds_ian, trues, False)


# df = pd.read_csv('input/bartley_sagittal_oof.csv')
# row_df = pd.read_csv('input/oof_predictions_submission_format_ip_v4_sub.csv')
# df = row_df.merge(df, on='row_id', how='left')
# df.loc[~df.normal_mild_y.isnull(), 'normal_mild'] = df.loc[~df.normal_mild_y.isnull(), 'normal_mild_y']
# df.loc[df.normal_mild_y.isnull(), 'normal_mild'] = df.loc[df.normal_mild_y.isnull(), 'normal_mild_x']
# df.loc[~df.moderate_y.isnull(), 'moderate'] = df.loc[~df.moderate_y.isnull(), 'moderate_y']
# df.loc[df.moderate_y.isnull(), 'moderate'] = df.loc[df.moderate_y.isnull(), 'moderate_x']
# df.loc[~df.severe_y.isnull(), 'severe'] = df.loc[~df.severe_y.isnull(), 'severe_y']
# df.loc[df.severe_y.isnull(), 'severe'] = df.loc[df.severe_y.isnull(), 'severe_x']
# df = df[list(row_df)]
# df['study_id'] = df.row_id.apply(lambda x: int(x.split('_')[0]))

# df['target_level'] = df.row_id.apply(lambda x: '_'.join(x.split('_')[1:]))
# df['target'] = df.row_id.apply(lambda x: '_'.join(x.split('_')[1:-2]))
# df['level'] = df.row_id.apply(lambda x: '_'.join(x.split('_')[-2:]))
# cols = ['spinal_canal_stenosis_l1_l2', 'spinal_canal_stenosis_l2_l3', 'spinal_canal_stenosis_l3_l4', 'spinal_canal_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1', 'left_neural_foraminal_narrowing_l1_l2', 'left_neural_foraminal_narrowing_l2_l3', 'left_neural_foraminal_narrowing_l3_l4', 'left_neural_foraminal_narrowing_l4_l5', 'left_neural_foraminal_narrowing_l5_s1', 'right_neural_foraminal_narrowing_l1_l2', 'right_neural_foraminal_narrowing_l2_l3', 'right_neural_foraminal_narrowing_l3_l4', 'right_neural_foraminal_narrowing_l4_l5', 'right_neural_foraminal_narrowing_l5_s1', 'left_subarticular_stenosis_l1_l2', 'left_subarticular_stenosis_l2_l3', 'left_subarticular_stenosis_l3_l4', 'left_subarticular_stenosis_l4_l5', 'left_subarticular_stenosis_l5_s1', 'right_subarticular_stenosis_l1_l2', 'right_subarticular_stenosis_l2_l3', 'right_subarticular_stenosis_l3_l4', 'right_subarticular_stenosis_l4_l5', 'right_subarticular_stenosis_l5_s1']

# m = {'study_id': []}
# for c in cols:
#     m[f'bartley_{c}_normal'] = []
#     m[f'bartley_{c}_moderate'] = []
#     m[f'bartley_{c}_severe'] = []
# for i, idf in df.groupby('study_id'):
#     m['study_id'].append(i)

#     for tl, ldf in idf.groupby('target_level'):
#         m[f'bartley_{tl}_normal'].append(ldf.normal_mild.values[0])
#         m[f'bartley_{tl}_moderate'].append(ldf.moderate.values[0])
#         m[f'bartley_{tl}_severe'].append(ldf.severe.values[0])
# bartley = pd.DataFrame(m).sort_values('study_id')
# preds_bartley = torch.FloatTensor(bartley[[c.replace('pred_', 'bartley_') for c in pred_cols]].fillna(0).values)
# trues = torch.FloatTensor(oof[[c.replace('pred_', '') for c in pred_cols]].fillna(0).values.astype(int))
# cri(preds_bartley, trues, False)


preds.numpy().shape, len(cols)


th = 0


# ④ 計算每個樣本欄位的 loss 差異（|真實值 - 預測值|）
# preds = (preds_yuji*2+preds_ian*2+preds_bartley)/5
preds = (preds_yuji*2)/2

# 對 spinal 這 5 個 level 的 severe (索引2,5,8,11,14) 進行強化
spinal = preds.numpy()[:, [2,5,8,11,14]]
new_preds = []
for v in spinal:
    i = v.tolist().index(v.max())
    if v.max() > th:  # th = 0 (永遠成立)
        v[i]*=1.25    # 對 spinal severe 五個位置中機率最大的那個，無條件放大 1.25 倍 -> (局部額外加強)
    new_preds.append(v)
preds[:, [2,5,8,11,14]] = torch.FloatTensor(new_preds)

# 重新對每個 group(三類 normal/moderate/severe) 做 normalize，確保三類加總為 1
for i in range(5):
    preds[:, i*3:(i+1)*3] = normalize_probabilities_to_one_torch(preds[:, i*3:(i+1)*3])
oof[pred_cols] = preds.numpy()
cri(preds, trues, False)

# 計算 loss 差異 (預測誤差)
for c in [c.replace('pred_', '') for c in pred_cols]:   # pred_cols = 75 個 各個病狀的嚴重程度
    oof[f'{c}_loss'] = np.abs(oof[c].values-oof['pred_'+c].values)
# oof[['study_id']+pred_cols+[c.replace('pred_', '')+'_loss' for c in pred_cols]].to_csv('results/oof_ensemble.csv', index=False)
oof[['study_id']+pred_cols+[c.replace('pred_', '')+'_loss' for c in pred_cols]].to_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/oof_ensemble.csv', index=False)  # pred_cols(共 75 欄)、xxx_loss(也是 75 欄)
oof[['study_id']+pred_cols+[c.replace('pred_', '')+'_loss' for c in pred_cols]]
oof.to_csv('oof6.csv')  # 我加 


# ⑤-②  用 confusion matrix 的方式
import numpy as np
import pandas as pd

# pred_cols = 你的預測欄位（應為 75 個，以 pred_ 開頭）
# 比如：
# pred_cols = [c for c in oof.columns if c.startswith('pred_') and c.endswith(('normal','moderate','severe'))]

noisy_rows = []
for i in range(0, len(pred_cols), 3):
    pred_col_group = pred_cols[i:i+3]  # pred_cols = 75 個 各個病狀的嚴重程度
    true_col_group = [c.replace('pred_', '') for c in pred_col_group]

    # 防呆：若真實欄位不存在就跳過
    if not all([c in oof.columns for c in true_col_group]):
        print(f"缺少真實標籤欄位：{true_col_group}")
        continue

    pred_prob = oof[pred_col_group].values  # 機率預測
    true_onehot = oof[true_col_group].values.astype(int)  # one-hot 標籤

    pred_label = np.argmax(pred_prob, axis=1)
    true_label = np.argmax(true_onehot, axis=1)
    is_noisy = pred_label != true_label
    noisy_idx = np.where(is_noisy)[0]

    # 找出疾病名/level
    base_col = pred_col_group[0].replace('pred_', '').replace('_normal', '')
    level = base_col.split('_')[-2] + '_' + base_col.split('_')[-1]
    target = '_'.join(base_col.split('_')[:-2])

    for idx in noisy_idx:
        study_id = oof.iloc[idx]['study_id']
        noisy_rows.append({
            'study_id': study_id,
            'target': target,
            'level': level,
            'row_idx': idx,
        })

noise_df = pd.DataFrame(noisy_rows)
noise_df['study_level'] = noise_df['study_id'].astype(str) + '_' + noise_df['level']
noise_df = noise_df.sort_values(['target', 'study_id', 'level'])
noise_df.to_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_confusion.csv', index=False)

import numpy as np
import pandas as pd
from collections import defaultdict

# 記得 pred_cols 必須是有順序的一組一組，每3欄一個 group
# 這裡直接先產生所有類別的統計
result = defaultdict(lambda: defaultdict(int))  # result[group][class] = [TP,TN,FP,FN]

for i in range(0, len(pred_cols), 3):
    pred_col_group = pred_cols[i:i+3]
    true_col_group = [c.replace('pred_', '') for c in pred_col_group]

    base_col = pred_col_group[0].replace('pred_', '').replace('_normal', '')
    class_names = ['normal', 'moderate', 'severe']

    pred_prob = oof[pred_col_group].values
    true_onehot = oof[true_col_group].values.astype(int)
    pred_label = np.argmax(pred_prob, axis=1)
    true_label = np.argmax(true_onehot, axis=1)

    for class_idx, class_name in enumerate(class_names):
        # binary: 以 class_idx 為正類，其他為負
        pred_bin = (pred_label == class_idx)
        true_bin = (true_label == class_idx)
        TP = np.logical_and(pred_bin, true_bin).sum()
        TN = np.logical_and(~pred_bin, ~true_bin).sum()
        FP = np.logical_and(pred_bin, ~true_bin).sum()
        FN = np.logical_and(~pred_bin, true_bin).sum()
        result[base_col+'_'+class_name]['TP'] += TP
        result[base_col+'_'+class_name]['TN'] += TN
        result[base_col+'_'+class_name]['FP'] += FP
        result[base_col+'_'+class_name]['FN'] += FN

# 最後統計出來（以全部的 sum 為例，也可細分 group）
TP = sum(v['TP'] for v in result.values())
TN = sum(v['TN'] for v in result.values())
FP = sum(v['FP'] for v in result.values())
FN = sum(v['FN'] for v in result.values())

print(f"TP: {TP}  TN: {TN}  FP: {FP}  FN: {FN}")

# 如果想看單一 group 下某一類別的 TP/TN/FP/FN：
# print(result['spinal_canal_stenosis_l1_l2_normal'])




# ⑤ 輸出含有 noisy 預測的樣本（th=0.8）
th = 0.8
m = {}
levels = []
cols = []
ids = []
for col in ['spinal_canal_stenosis','left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing', 'left_subarticular_stenosis', 'right_subarticular_stenosis']:
    # oof = pd.read_csv('results/oof_ensemble.csv')
    oof = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/oof_ensemble.csv')
    for level_i, level in zip([1,2,3,4,5], ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']):
        dfs = []
        for claz in ['normal', 'moderate', 'severe']:
            dfs.append(oof[oof[f'{col}_{level}_{claz}_loss'] > th])
        noisy_df = pd.concat(dfs)
        for id in noisy_df.study_id.unique():
            levels.append(level)
            cols.append(col)
            ids.append(id)

noise_df = pd.DataFrame({
    'study_id': ids,
    'target': cols,
    'level': levels,
}).sort_values(['target','study_id','level'])
noise_df['study_level'] = noise_df.study_id.astype(str) + '_' + noise_df.level

# noise_df.to_csv(f'results/noisy_target_level_th08.csv', index=False)
noise_df.to_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th08.csv', index=False)
# print(th, noise_df.target.value_counts().values / (1975*5)) 
print(th, noise_df.target.value_counts().values / (393*5))  # 得到 5 個不同 col 的數值 ['spinal_canal_stenosis','left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing', 'left_subarticular_stenosis', 'right_subarticular_stenosis']
noise_df.target.value_counts()


# ⑥ 輸出 noisy 樣本（th=0.9）
th = 0.9
m = {}
levels = []
cols = []
ids = []
for col in ['spinal_canal_stenosis','left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing', 'left_subarticular_stenosis', 'right_subarticular_stenosis']:
    # oof = pd.read_csv('results/oof_ensemble.csv')
    oof = pd.read_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/oof_ensemble.csv')
    for level_i, level in zip([1,2,3,4,5], ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']):
        dfs = []
        for claz in ['normal', 'moderate', 'severe']:
            dfs.append(oof[oof[f'{col}_{level}_{claz}_loss'] > th])
        noisy_df = pd.concat(dfs)
        for id in noisy_df.study_id.unique():
            levels.append(level)
            cols.append(col)
            ids.append(id)

noise_df = pd.DataFrame({
    'study_id': ids,
    'target': cols,
    'level': levels,
}).sort_values(['target','study_id','level'])
noise_df['study_level'] = noise_df.study_id.astype(str) + '_' + noise_df.level

# noise_df.to_csv(f'results/noisy_target_level_th09.csv', index=False)
noise_df.to_csv(f'{WORKING_DIR}/csv_train/noise_reduction_by_oof_9/noisy_target_level_th09.csv', index=False)
# print(th, noise_df.target.value_counts().values / (1975*5))
print(th, noise_df.target.value_counts().values / (393*5))
