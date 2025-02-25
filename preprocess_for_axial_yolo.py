import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

# kaggle input
DATA_KAGGLE_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"

# 設定環境變數
WORKING_DIR="/kaggle/working/duplicate"

# train = pd.read_csv('input/train_with_fold.csv')
train = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_4/train_with_fold.csv')
train = train[train.series_description_x=='Axial T2']
train['instance_number'] = train.path.apply(lambda x: int(x.split('___')[-1].replace('.png', '')))

dfs = []
for id, idf in train.groupby('series_id'):
    idf = idf.sort_values(['instance_number', 'z_pos'])
    idf = idf.drop_duplicates('z_pos')
    dfs.append(idf)
train = pd.concat(dfs)

# cood = pd.read_csv('input/train_label_coordinates.csv')
cood = pd.read_csv(f'{DATA_KAGGLE_DIR}/train_label_coordinates.csv')
cood = cood.sort_values(['series_id', 'instance_number'])
train = train.merge(cood, on=['study_id', 'series_id','instance_number'])

train['class_id'] = train.level.map({
    'L1/L2': 0,
    'L2/L3': 1,
    'L3/L4': 2,
    'L4/L5': 3,
    'L5/S1': 4,
})
train['class_name'] = train.level.values
train['x_min'] = np.round(train['x'].values-10).astype(int)
train['y_min'] = np.round(train['y'].values-10).astype(int)
train['x_max'] = np.round(train['x'].values+10).astype(int)
train['y_max'] = np.round(train['y'].values+10).astype(int)

import cv2
from multiprocessing import Pool, cpu_count

def exec(p):
    im=cv2.imread(p)
    return im.shape[:2]

p = Pool(processes=4)
results = []
args = train.path.values
with tqdm(total=len(args)) as pbar:
    for res in p.imap(exec, args):
        results.append(res)
        pbar.update(1)
p.close()
train[['image_height', 'image_width']] = np.array(results)
train['x_min'] = np.round(train['x'].values-train['image_width'].values/30).astype(int)
train['y_min'] = np.round(train['y'].values-train['image_height'].values/30).astype(int)
train['x_max'] = np.round(train['x'].values+train['image_width'].values/30).astype(int)
train['y_max'] = np.round(train['y'].values+train['image_height'].values/30).astype(int)

# train.to_csv('input/train_axial_for_yolo_all_image_v1.csv', index=False)
train.to_csv('train_axial_for_yolo_all_image_v1.csv', index=False)
