import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from multiprocessing import Pool, cpu_count

# kaggle input
DATA_KAGGLE_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"

# 設定環境變數
WORKING_DIR="/kaggle/working/duplicate"

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# df = pd.read_csv('input/train_with_fold.csv')
df = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_4/train_with_fold.csv')
df['instance_number'] = df.path.apply(lambda x: int(x.split('___')[-1].replace('.png', '')))
dfs = []
for id, idf in df.groupby('series_id'):
    idf = idf.sort_values(['x_pos', 'instance_number'])
    idf = idf.drop_duplicates('x_pos')
    ldf = idf.iloc[len(idf)//2:len(idf)//2+1]
    dfs.append(ldf)
df = pd.concat(dfs)

# coords = pd.read_csv('input/coords_rsna_improved.csv')
coords = pd.read_csv(f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_6/coords_rsna_improved.csv')
coords['class_name'] = coords['level'] + '_' + coords.side
coords[['study_id', 'relative_x', 'relative_y', 'series_id', 'instance_number', 'class_name']].sort_values(['study_id', 'series_id', 'instance_number']).tail(20)
coords = coords[["series_id", "class_name", "relative_x", "relative_y"]]

def exec(p):
    im=cv2.imread(p)
    return im.shape[:2]

p = Pool(processes=4)
results = []
args = df.path.values
with tqdm(total=len(args)) as pbar:
    for res in p.imap(exec, args):
        results.append(res)
        pbar.update(1)
p.close()

df[['image_height', 'image_width']] = np.array(results)
df = df.merge(coords, on='series_id')

df['x'] = df['relative_x'] * df['image_width']
df['y'] = df['relative_y'] * df['image_height']
df['x_min'] = np.round(df['x'].values-df['image_width'].values/30).astype(int)
df['y_min'] = np.round(df['y'].values-df['image_height'].values/30).astype(int)
df['x_max'] = np.round(df['x'].values+df['image_width'].values/30).astype(int)
df['y_max'] = np.round(df['y'].values+df['image_height'].values/30).astype(int)

from sklearn import preprocessing

df=df.sort_values('class_name')
label_encoder = preprocessing.LabelEncoder()
col = 'class_name'
df['class_id'] = label_encoder.fit_transform(df[col])
df = df.sort_values(['series_id', 'class_id'])
df = df[df.series_description == "Sagittal T2/STIR"]
# df.to_csv('input/train_for_yolo_10level_v1.csv', index=False)
df.to_csv(f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_6/train_for_yolo_10level_v1.csv', index=False)

targets = ['l1_spinal', 'l2_spinal', 'l3_spinal', 'l4_spinal', 'l5_spinal', 'l1_right_neural', 'l2_right_neural', 'l3_right_neural', 'l4_right_neural', 'l5_right_neural', 'l1_left_neural', 'l2_left_neural', 'l3_left_neural', 'l4_left_neural', 'l5_left_neural']
targets = [f'pred_{c}' for c in targets]
pred_cols = [f'pred_{c}' for c in targets]

# oof = pd.concat([pd.read_csv(f'results/rsna_sagittal_cl/oof_fold{fold}.csv') for fold in range(5)])
oof = pd.concat([pd.read_csv(f'{WORKING_DIR}/ckpt/rsna_sagittal_cl/oof_fold{fold}.csv') for fold in range(1)])
oof[pred_cols] = sigmoid(oof[pred_cols])
oof['pred_spinal'] = oof[[c for c in pred_cols if 'spinal' in c]].mean(1)
oof['pred_right_neural'] = oof[[c for c in pred_cols if 'right_neural' in c]].mean(1)
oof['pred_left_neural'] = oof[[c for c in pred_cols if 'left_neural' in c]].mean(1)
# oof.to_csv('results/rsna_sagittal_cl/oof.csv', index=False)
oof.to_csv(f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_6/rsna_sagittal_cl/oof.csv', index=False)