# wbf.py (且去除掉重複和低信心預測)
from src.utils.ensemble_boxes import *  # weighted_boxes_fusion 我猜
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
warnings.simplefilter('ignore')
from multiprocessing import cpu_count

# 在 kaggle 執行時使用
# wbf_path = "/kaggle/working/duplicate"
# !python3 {wbf_path}/wbf.py --configs rsna_10classes_yolox_x

# kaggle input
DATA_KAGGLE_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"

# 設定環境變數
WORKING_DIR="/kaggle/working/duplicate"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--configs", '-c', nargs='+', type=str, default=['test'])
# parser.add_argument("--configs", '-c', nargs='+', type=str, default=['rsna_axial_all_images_left_yolox_x'])
# parser.add_argument('--fold', '-f', nargs='+', type=int, default=[0,1,2,3,4])
parser.add_argument('--fold', '-f', nargs='+', type=int, default=[1])  # for rsna_10classes_yolox_x
# parser.add_argument('--fold', '-f', nargs='+', type=int, default=[0])  # for rsna_axial_all_images_left_yolox_x\rsna_axial_all_images_right_yolox_x
args = parser.parse_args()

configs = args.configs

from src.yolo_configs import *
cfg = eval(configs[0])()  # configs = {rsna_axial_all_images_left_yolox_x}
# 假設你有一個字典或模塊，其中包含所有類
# class_map = {
#     "rsna_axial_all_images_left_yolox_x": rsna_axial_all_images_left_yolox_x,
#     "rsna_axial_all_images_right_yolox_x": rsna_axial_all_images_right_yolox_x,
#     "rsna_10classes_yolox_x": rsna_10classes_yolox_x
# }
# 根據 configs[0] 來獲取類並創建實例
# cfg_class = class_map[configs[0]]  # 根據配置獲取對應的類
# cfg = cfg_class()  # 創建該類的實例
# print(cfg)

tests = []
for model_n, config in enumerate(configs):
    for fold in args.fold:  # default=[0,1]
        # test = pd.read_csv(f'results/{config}/test_fold{fold}.csv')
        test = pd.read_csv(f'{WORKING_DIR}/results/{config}/test_fold{fold}.csv')  # rsna_axial_all_images_left_yolox_x/test_fold0.csv (sagittal region estimation)
        print('configs = ', config)  # 我加
        test['model_n'] = f'{model_n}_fold{fold}'  # 1_fold0、1_fold1
        tests.append(test)
test = pd.concat(tests)
box_cols = ['x_min', 'y_min', 'x_max', 'y_max']
weights = [1]*len(tests)
iou_thr = 0.5
skip_box_thr = 0.0001
results = []
max_value = 12800

from multiprocessing import Pool

def exec(args):
    path, path_df = args
    boxes_list = []
    confs_list = []
    labels_list = []
    for _, model_df in path_df.groupby('model_n'):
        boxes_list.append(model_df[box_cols].values/max_value)
        confs_list.append(model_df['conf'].values.tolist())
        labels_list.append(model_df['class_id'].values.tolist())
    boxes, confs, labels = weighted_boxes_fusion(boxes_list, confs_list, labels_list, weights=[1]*len(boxes_list), iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes *= max_value
    results = []
    for idx, box in enumerate(boxes):
        results.append({
            "path": path,
            # "fold": fold,
            "class_id": int(labels[idx]),
            'conf':confs[idx],
            "x_min": box[0],
            "y_min": box[1],
            "x_max": box[2],
            "y_max": box[3],
        })
    return results

wbf_result_maps_list = []
df_list = list(test.groupby('path'))
p = Pool(processes=cpu_count())
with tqdm(total=len(df_list)) as pbar:
    for wbf_result_maps in p.imap(exec, df_list):
        wbf_result_maps_list += wbf_result_maps
        pbar.update(1)
p.close()

results = pd.DataFrame(wbf_result_maps_list)

# os.makedirs(f'results/wbf', exist_ok=True)
os.makedirs(f'{WORKING_DIR}/results/wbf', exist_ok=True)
print('='*100)
filename = "_".join(configs)
print(f"pd.read_csv(f\'{WORKING_DIR}/results/wbf/{filename}.csv\')")
print('='*100)
results.to_csv(f'{WORKING_DIR}/results/wbf/{"_".join(configs)}.csv', index=False)
