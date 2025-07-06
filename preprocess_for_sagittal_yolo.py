# preprocess_for_sagittal_yolo.py
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
# df = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_4/train_with_fold.csv')
df = pd.read_csv(f'{WORKING_DIR}/csv_train/preprocess_holdout_4/train_with_fold_holdout.csv')
df['instance_number'] = df.path.apply(lambda x: int(x.split('___')[-1].replace('.png', '')))
dfs = []
for id, idf in df.groupby('series_id'):
    idf = idf.sort_values(['x_pos', 'instance_number'])
    idf = idf.drop_duplicates('x_pos')
    ldf = idf.iloc[len(idf)//2:len(idf)//2+1]  # 除法取整數 (選到中間以及中間+1的 slice)
    dfs.append(ldf)  # 對一個 series_id 的 group
df = pd.concat(dfs)  # 對所有 series_id 的 group

# coords = pd.read_csv('input/coords_rsna_improved.csv')
coords = pd.read_csv(f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_6/coords_rsna_improved.csv')
coords['class_name'] = coords['level'] + '_' + coords.side  # 新增欄位 L1/L2_L or L1/L2_R
coords[['study_id', 'relative_x', 'relative_y', 'series_id', 'instance_number', 'class_name']].sort_values(['study_id', 'series_id', 'instance_number']).tail(20)  # tail(20) 只印出排序後最後 20 筆資料
coords = coords[["series_id", "class_name", "relative_x", "relative_y"]]  # 篩選所需欄位

def exec(p):
    im=cv2.imread(p)
    return im.shape[:2]  # image 的 (height, width, channels) 回傳前兩個

p = Pool(processes=4)
results = []
args = df.path.values  # 以 numpy.ndarray(N-dimensional array) 的方式回傳
with tqdm(total=len(args)) as pbar:
    for res in p.imap(exec, args):  # 回傳得到 (height, width)
        results.append(res)
        pbar.update(1)
p.close()

df[['image_height', 'image_width']] = np.array(results)
df = df.merge(coords, on='series_id')

df['x'] = df['relative_x'] * df['image_width']  # coords_rsna_improved.csv 的用途，像是特徵的定位點(x, y) -> 會隨著比例的縮放
df['y'] = df['relative_y'] * df['image_height']
df['x_min'] = np.round(df['x'].values-df['image_width'].values/30).astype(int)
df['y_min'] = np.round(df['y'].values-df['image_height'].values/30).astype(int)
df['x_max'] = np.round(df['x'].values+df['image_width'].values/30).astype(int)
df['y_max'] = np.round(df['y'].values+df['image_height'].values/30).astype(int)

from sklearn import preprocessing

df=df.sort_values('class_name')
label_encoder = preprocessing.LabelEncoder()  # 它會自動找出所有出現過的類別（例如：L1_R, L2_L, L3_R），並幫每個類別編上一個數字（0, 1, 2...）
col = 'class_name'
df['class_id'] = label_encoder.fit_transform(df[col])  # 將 class_name 自動編碼到 class_id
df = df.sort_values(['series_id', 'class_id'])  # 以 'series_id', 'class_id' 作為主要排序
df = df[df.series_description_y == "Sagittal T2/STIR"]  # NFN
# df.to_csv('input/train_for_yolo_10level_v1.csv', index=False)
df.to_csv(f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_6/train_for_yolo_10level_v1.csv', index=False)

targets = ['l1_spinal', 'l2_spinal', 'l3_spinal', 'l4_spinal', 'l5_spinal', 'l1_right_neural', 'l2_right_neural', 'l3_right_neural', 'l4_right_neural', 'l5_right_neural', 'l1_left_neural', 'l2_left_neural', 'l3_left_neural', 'l4_left_neural', 'l5_left_neural']
targets = [f'pred_{c}' for c in targets]  # pred_l1_spinal (沒有做 sigmoid)
pred_cols = [f'pred_{c}' for c in targets]  # pred_pred_l1_spinal (有做 sigmoid)

# oof = pd.concat([pd.read_csv(f'results/rsna_sagittal_cl/oof_fold{fold}.csv') for fold in range(5)])
oof = pd.concat([pd.read_csv(f'{WORKING_DIR}/ckpt/rsna_sagittal_cl/oof_fold{fold}.csv') for fold in range(1)])  # 在 slice estimation 最後得到各個類別的分數

oof[pred_cols] = sigmoid(oof[pred_cols])
oof['pred_spinal'] = oof[[c for c in pred_cols if 'spinal' in c]].mean(1)  # 用有做 sigmoid 的結果去平均
oof['pred_right_neural'] = oof[[c for c in pred_cols if 'right_neural' in c]].mean(1)
oof['pred_left_neural'] = oof[[c for c in pred_cols if 'left_neural' in c]].mean(1)
# oof.to_csv('results/rsna_sagittal_cl/oof.csv', index=False)
oof.to_csv(f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_6/oof.csv', index=False)
print('preprocess_for_sagittal_yolo.py finish')

# train_for_yolo_10level_v1.csv 找出 bounding box 的位置
# oof.csv 找出 每個類別的各自的分數(包含是哪個 slice 的資訊)
