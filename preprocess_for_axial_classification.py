import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv('input/axial_closest_df.csv')
df['pred_level'] = df.level.values
df = df[df.dis < 3]
df = df[df.closest == 1][['series_id', 'instance_number', 'pred_level', 'dis']]
al = pd.read_csv('input/train_with_fold.csv')

tr = al.merge(df, on=['series_id', 'instance_number'])

label_features = [
    'spinal_canal_stenosis',
    'left_neural_foraminal_narrowing',
    'right_neural_foraminal_narrowing',
    'left_subarticular_stenosis',
    'right_subarticular_stenosis',
]
for col in label_features:
    tr[f'{col}_normal'] = 0
    tr[f'{col}_moderate'] = 0
    tr[f'{col}_severe'] = 0
    for level_n, level in enumerate(['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']):
        tr.loc[((tr[col+'_'+level]=='Normal/Mild') & (tr['pred_level']==level_n+1)), f'{col}_normal'] = 1
        tr.loc[((tr[col+'_'+level]=='Moderate') & (tr['pred_level']==level_n+1)), f'{col}_moderate'] = 1
        tr.loc[((tr[col+'_'+level]=='Severe') & (tr['pred_level']==level_n+1)), f'{col}_severe'] = 1


# axial right
test = pd.read_csv('results/rsna_axial_all_images_right_yolox_x/test_fold0.csv')
dfs=[]
for p, pdf in tqdm(test.groupby(["path", 'class_id'])):
    dfs.append(pdf[pdf.conf==pdf.conf.max()])
right=pd.concat(dfs)    
for c in ['conf', 'x_min', 'y_min', 'x_max', 'y_max']:
    right = right.rename(columns={c: 'right_'+c})

# axial left
test = pd.read_csv('results/rsna_axial_all_images_left_yolox_x/test_fold0.csv')
dfs=[]
for p, pdf in tqdm(test.groupby(["path", 'class_id'])):
    dfs.append(pdf[pdf.conf==pdf.conf.max()])
left = pd.concat(dfs)    
for c in ['conf', 'x_min', 'y_min', 'x_max', 'y_max']:
    left = left.rename(columns={c: 'left_'+c})
df = right.merge(left[['path']+['left_conf', 'left_x_min', 'left_y_min', 'left_x_max', 'left_y_max']], on='path')
df['x_min'] = df[['right_x_min', 'left_x_min']].min(1)
df['y_min'] = df[['right_y_min', 'left_y_min']].min(1)
df['x_max'] = df[['right_x_max', 'left_x_max']].max(1)
df['y_max'] = df[['right_y_max', 'left_y_max']].max(1)
df.to_csv('results/axial_yolo_results.csv', index=False)

boxdf = pd.read_csv('results/axial_yolo_results.csv')
boxdf = boxdf[['path','x_min', 'y_min', 'x_max', 'y_max']]
boxdf.path = boxdf.path.apply(lambda x: 'input/' + x.split('/input/')[-1])
train_df = tr.merge(boxdf, on='path')

import cv2
from multiprocessing import Pool, cpu_count

def exec(p):
    im=cv2.imread(p)
    return im.shape[:2]

p = Pool(processes=4)
results = []
args = train_df.path.values
with tqdm(total=len(args)) as pbar:
    for res in p.imap(exec, args):
        results.append(res)
        pbar.update(1)
p.close()
train_df[['image_height', 'image_width']] = np.array(results)

train_df.to_csv('input/axial_classification.csv', index=False)
print('save to input/axial_classification.csv')