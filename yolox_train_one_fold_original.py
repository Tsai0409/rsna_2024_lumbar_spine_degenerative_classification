# yolox_train_one_fold_original.py
import warnings
warnings.filterwarnings("ignore")
import os
import sys  # 我加
import json
import pandas as pd
import numpy as np
import cv2
import gc
from pdb import set_trace as st
from tqdm.notebook import tqdm
tqdm.pandas()
from sklearn.model_selection import GroupKFold
import sys
from src.yolo_configs import *

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_annot_json(json_annotation, filename):
    json.dump(json_annotation, open(filename, 'w'), indent=4, cls=NumpyEncoder)

annotion_id = 0
image_id_n = 0
def dataset2coco(df):
    global annotion_id
    global image_id_n
    annotations_json = {  # 建立 annotations_json 字典
        "info": [],
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }
    info = {
        "year": "2023",
        "version": "1",
        "description": f"{cfg.compe} dataset - COCO format",  # cfg.compe = 'rsna_2024'
        "contributor": "yujiariyasu",
        "url": "https://kaggle.com",
        "date_created": "2023-04-10T15:01:26+00:00"
    }
    annotations_json["info"].append(info)
    lic = {
            "id": 1,
            "url": "",
            "name": "Unknown"
        }
    annotations_json["licenses"].append(lic)
    for id_n, (path, idf) in enumerate(df.groupby('path')):
        images = {
            "id": image_id_n,
            "license": 1,
            "file_name": path,
            "height": idf.image_height.values[0],
            "width": idf.image_width.values[0],
            "date_captured": "2023-04-10T15:01:26+00:00"
        }

        annotations_json["images"].append(images)
        for _, row in idf.iterrows():
            bbox = row[['x_min', 'y_min', 'x_max', 'y_max']].values
            b_width = bbox[2]-bbox[0]
            b_height = bbox[3]-bbox[1]

            image_annotations = {
                "id": annotion_id,
                "image_id": image_id_n,
                "category_id": row.class_id,
                "bbox": [bbox[0], bbox[1], b_width, b_height],
                "area": b_width * b_height,
                "segmentation": [],
                "iscrowd": 0
            }

            annotion_id += 1
            annotations_json["annotations"].append(image_annotations)
        image_id_n += 1
    print(f"len(df): {len(df)}")
    return annotations_json

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("--config", '-c', type=str, default='Test', help="config name in configs.py")
parser.add_argument("--config", '-c', type=str, default='Test', help="config name in yolo_configs.py")
# parser.add_argument("--gpu", '-g', type=str, default='nochange', help="config name in configs.py")
parser.add_argument("--gpu", '-g', type=str, default='nochange', help="config name in yolo_configs.py")
# parser.add_argument("--fold", type=int, default=0, help="fold num")
parser.add_argument("--fold", '-f', type=int, default=0, help="fold num")
parser.add_argument("--use_row", type=int, default=2, help="google spread sheet row")
parser.add_argument("--make_labels", action='store_true', help="make_labels")

args = parser.parse_args()
print(args)
fold = args.fold
config = args.config
cfg = eval(args.config)()
# absolute_path = /kaggle/working/duplicate
print('absolute_path = '+cfg.absolute_path)
# cfg.train_df.path = cfg.absolute_path + '/' + cfg.train_df.path  # train 照片路徑；ex:/kaggle/temp/axial_all_images/2767326159___223384___5.png
cfg.train_df.path = cfg.train_df.path
# cfg.test_df.path = cfg.absolute_path + '/' + cfg.test_df.path
cfg.test_df.path = cfg.test_df.path

cfg.train_df.class_id = cfg.train_df.class_id.astype(int)
if 'x_min' not in list(cfg.train_df):
    cfg.train_df['x_min'] = cfg.train_df['image_width'] * (cfg.train_df['x_center_scaled']-cfg.train_df['width_scaled']/2)
if 'x_max' not in list(cfg.train_df):
    cfg.train_df['x_max'] = cfg.train_df['image_width'] * (cfg.train_df['x_center_scaled']+cfg.train_df['width_scaled']/2)
if 'y_min' not in list(cfg.train_df):
    cfg.train_df['y_min'] = cfg.train_df['image_height'] * (cfg.train_df['y_center_scaled']-cfg.train_df['height_scaled']/2)
if 'y_max' not in list(cfg.train_df):
    cfg.train_df['y_max'] = cfg.train_df['image_height'] * (cfg.train_df['y_center_scaled']+cfg.train_df['height_scaled']/2)

sys.path.append("/kaggle/working/duplicate/src/YOLOX")  # 我加
# os.chdir('src/YOLOX')
os.chdir('/kaggle/working/duplicate/src/YOLOX')

print(f'\n----------------------- Config -----------------------')
config_str = ''
for k, v in vars(cfg).items():
    if (k == 'model') | ('df' in k):
        continue
    print(f'\t{k}: {v}')
    config_str += f'{k}: {v}, '
print(f'----------------------- Config -----------------------\n')

# absolute_path = /kaggle/working/duplicate
# configs = rsna_axial_all_images_left_yolox_x、rsna_axial_all_images_right_yolox_x
config_path = f'configfile_{config}_fold{fold}.py'
os.makedirs(f'{cfg.absolute_path}/results/{config}', exist_ok=True)

if hasattr(cfg, 'model_name_for_yolox'):
    model_name = cfg.model_name_for_yolox
else:
    model_name = cfg.pretrained_path.split('/')[-1].replace('.pth', '')
categories = []
class_id_name_map = {}

for n, (c, id) in enumerate(zip(cfg.train_df.sort_values('class_id').class_name.unique(), cfg.train_df.sort_values('class_id').class_id.unique())):
    classes = {'supercategory': 'none'}
    classes['id'] = id
    classes['name'] = c
    categories.append(classes)
    class_id_name_map[id] = c
print('class_id_name_map:', class_id_name_map)
tr = cfg.train_df[cfg.train_df.fold != fold]
val = cfg.train_df[cfg.train_df.fold == fold]

# ====== 資料檢查：訓練/驗證集是否正常 ======
print("\n🛡 資料檢查開始...")

required_columns = ['path', 'x_min', 'y_min', 'x_max', 'y_max', 'class_id']
missing_cols = [col for col in required_columns if col not in cfg.train_df.columns]

if len(cfg.train_df) == 0:
    raise ValueError("❌ cfg.train_df 為空，請確認 CSV 是否正確讀入")

if missing_cols:
    raise ValueError(f"❌ 缺少必要欄位: {missing_cols}，請檢查 train_df 是否已經計算出 x_min/x_max 等欄位")

if cfg.train_df[['x_min', 'y_min', 'x_max', 'y_max']].isnull().any().any():
    raise ValueError("❌ 某些 bbox 欄位包含 NaN，可能是 scaled 座標轉換出錯")

if (cfg.train_df['x_max'] <= cfg.train_df['x_min']).any() or (cfg.train_df['y_max'] <= cfg.train_df['y_min']).any():
    raise ValueError("❌ 發現無效 bbox（x_max <= x_min 或 y_max <= y_min），請檢查標註格式")

if len(tr) == 0:
    raise ValueError("❌ 訓練資料筆數為 0，可能是 fold 設定錯誤")

if len(val) == 0:
    raise ValueError("❌ 驗證資料筆數為 0，請檢查 fold 是否切得太極端或資料太少")

print(f"✅ 資料正常！Train: {len(tr)} 筆, Val: {len(val)} 筆")
print("🛡 資料檢查結束。\n")

import cv2
from tqdm import tqdm

print("🖼 開始檢查圖片檔案是否存在與可讀取...")

# 只檢查部分圖片（例如 100 張）避免太慢
sample_paths = cfg.train_df['path'].dropna().unique()
sample_paths = sample_paths[:100]  # 或用 random.sample 更隨機

not_found = []
not_readable = []

for img_path in tqdm(sample_paths, desc="Checking images"):
    if not os.path.exists(img_path):
        not_found.append(img_path)
        continue

    img = cv2.imread(img_path)
    if img is None:
        not_readable.append(img_path)

if not_found:
    print(f"❌ 找不到圖片檔案數量：{len(not_found)}")
    print("範例路徑：", not_found[:3])
else:
    print("✅ 所有測試圖片都存在於磁碟上")

if not_readable:
    print(f"❌ 有 {len(not_readable)} 張圖片 cv2.imread 讀取失敗（回傳 None）")
    print("範例路徑：", not_readable[:3])
else:
    print("✅ 所有測試圖片 cv2.imread 讀取成功")

print("🖼 圖片檢查結束。\n")

# ✅ 1. 檢查驗證集是否有標註資料
# 檢查 val 是否真的有 bbox（確保 class_id 存在且 bbox 有意義）
print("🔍 驗證資料 bbox 檢查...")
print("val bbox count:", len(val))
print("val 中不同 class 數量:", val['class_id'].nunique())
print("val 中各 class 數量:")
print(val['class_id'].value_counts())

# 若要更詳細看是否有 bbox 無效
invalid_bbox = val[(val['x_max'] <= val['x_min']) | (val['y_max'] <= val['y_min'])]
print("❗ 無效 bbox 數量：", len(invalid_bbox))

# ✅ 2. 檢查 cfg.predict_valid 是否為 True
print("cfg.predict_valid =", cfg.predict_valid)

# 


print('len(train) / len(val):', len(tr), len(val))
# self.train_df_path = f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_6/train_axial_for_yolo_all_image_v1.csv'
train_df_filename = args.config + '___' + cfg.train_df_path.split('/')[-1].replace('.csv', '')  # rsna_axial_all_images_left_yolox_x___train_axial_for_yolo_all_image_v1
train_json_filename = f'train_{train_df_filename}_fold{fold}_len{len(tr)}.json'  # train_rsna_axial_all_images_left_yolox_x___train_axial_for_yolo_all_image_v1_fold0_len .json
valid_json_filename = f'valid_{train_df_filename}_fold{fold}_len{len(val)}.json'

# class rsna_axial_all_images_left_yolox_x、class rsna_axial_all_images_right_yolox_x 的 cfg.inference_only=False
# class rsna_10classes_yolox_x 的 cfg.inference_only=True
if not cfg.inference_only:
    if os.path.exists(f"{cfg.absolute_path}/input/annotations/{train_json_filename}") & os.path.exists(f"{cfg.absolute_path}/input/annotations/{valid_json_filename}") & (not cfg.update_json):
        print('make labels skip.')
    else:
        print('make labels start...')
        train_annot_json = dataset2coco(tr)  # 會用到 cfg.train_df.path = cfg.absolute_path + '/' + cfg.train_df.path；可能有錯？
        valid_annot_json = dataset2coco(val)
        os.system(f'mkdir -p {cfg.absolute_path}/input/annotations/')
        save_annot_json(train_annot_json, f"{cfg.absolute_path}/input/annotations/{train_json_filename}")
        save_annot_json(valid_annot_json, f"{cfg.absolute_path}/input/annotations/{valid_json_filename}")

config_file_template = f'''

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import sys

# 設定 PYTHONPATH 確保能夠找到 yolox 模組 我加
sys.path.append("/kaggle/working/duplicate/src/YOLOX")
os.chdir('/kaggle/working/duplicate/src/YOLOX')
os.environ["PYTHONPATH"] = "/kaggle/working/duplicate/src/YOLOX:" + os.environ.get("PYTHONPATH", "")

# 設定訓練命令 我加
train_str = f'PYTHONPATH=/kaggle/working/duplicate/src/YOLOX python tools/train.py -f configfile_rsna_axial_all_images_left_yolox_x_fold0.py -d 1 -b 8 --fp16 -o -c /groups/gca50041/ariyasu/yolox_weights/yolox_x.pth'

from yolox.exp import Exp as MyExp  # 用到 YOLOX/yolox

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        if '{model_name}' == 'yolox_s':
            self.depth = 0.33
            self.width = 0.50
        elif '{model_name}' == 'yolox_m':
            self.depth = 0.67
            self.width = 0.75
        elif '{model_name}' == 'yolox_l':
            self.depth = 1.0
            self.width = 1.0
        elif '{model_name}' == 'yolox_x':
            self.depth = 1.33
            self.width = 1.25
        else:
            raise
        self.exp_name = '{config}'
        self.data_dir = ""

        ### need change ###
        self.max_epoch = {cfg.epochs}
        self.train_ann = "{cfg.absolute_path}/input/annotations/{train_json_filename}"
        self.val_ann = "{cfg.absolute_path}/input/annotations/{valid_json_filename}"
        self.output_dir = "{cfg.absolute_path}/results/{config}/fold{fold}"  # absolute_path = /kaggle/working/duplicate
        self.input_size = {cfg.image_size}
        self.test_size = {cfg.image_size}
        self.no_aug_epochs = {cfg.no_aug_epochs} # 15
        self.warmup_epochs = {cfg.warmup_epochs} # 5
        self.num_classes = {cfg.train_df.class_name.nunique()}
        self.categories = {categories}
        self.class_id_name_map = {class_id_name_map}
        ### need change ###

        ### ✅ 新增這兩行 ###
        self.save_history_ckpt = True
        self.test_conf = 0.001

        ### fyi ###
        self.data_num_workers = {cfg.batch_size}
        self.eval_interval = 1
        self.seed = 42
        self.print_interval = 100
        self.eval_interval = 1
        self.save_history_ckpt = False
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.min_lr_ratio = 0.05
        self.basic_lr_per_img = 0.00015625
        self.scheduler = 'yoloxwarmcos'
        self.ema = True
        self.weight_decay = 0.0005
        self.momentum = 0.9
        self.test_conf = 0.01
        self.nmsthre = {cfg.nmsthre}
        ### fyi ###

        if {cfg.heavy_aug}:
            self.scale = (0.1, 2)
            self.mosaic_scale = (0.8, 1.6)
            self.perspective = 0.0

'''

with open(config_path, 'w') as f:
    f.write(config_file_template)

from pycocotools.coco import COCO
from random import sample
import importlib

# class rsna_axial_all_images_left_yolox_x、class rsna_axial_all_images_right_yolox_x 的 cfg.inference_only=False
# class rsna_10classes_yolox_x 的 cfg.inference_only=True
if cfg.inference_only:
    print('inference_only.')
else:  # here
    print('train start...')
    # train_str = f'python train.py -f {config_path} -d 1 -b {cfg.batch_size} --fp16 -o -c {cfg.pretrained_path}'
    train_str = f'python tools/train.py -f {config_path} -d 1 -b {cfg.batch_size} --fp16 -o -c {cfg.pretrained_path}'

    # class Baseline: cfg.resume = False
    if cfg.resume:  # no here
        train_str = f'python train.py -f {config_path} -d 1 -b {cfg.batch_size} --fp16 -o -c {cfg.absolute_path}/results/{config}/fold{fold}/{config}/best_ckpt.pth --resume --start_epoch {cfg.resume_start_epoch}'

    print('train_str:', train_str)  # train_str: python tools/train.py -f configfile_rsna_axial_all_images_left_yolox_x_fold0.py -d 1 -b 8 --fp16 -o -c /kaggle/input/pretrain-7/yolox_x.pth
    os.system(train_str)

### inference ###
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count
# sys.path.append('')
sys.path.append(f'{cfg.absolute_path}/src/YOLOX')  # /kaggle/working/duplicate/src/YOLOX

from yolox.utils import postprocess
from yolox.data.data_augment import ValTransform
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class MyDataset(Dataset):
    def __init__(self, cfg, df):
        self.paths = df.path.unique()
        self.cfg = cfg
        self.preproc = ValTransform(legacy = False)

    def __len__(self):
        return len(self.paths)

    def _read_image(self, path):
        if '.npy' in path:
            image = np.load(path)
            image = np.array([image, image, image]).transpose((1,2,0))
        else:
            image = cv2.imread(path)

        return image

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = self._read_image(path)
        ratio = min(cfg.image_size[0] / img.shape[0], cfg.image_size[1] / img.shape[1])
#         ratio = min(512 / img.shape[0], 512 / img.shape[1])

        img, _ = self.preproc(img, None, cfg.image_size)
        img = torch.from_numpy(img).float()
        img = img.float()

        return img, path, ratio

# get YOLOX experiment

current_exp = importlib.import_module(config_path.replace('.py', ''))
exp = current_exp.Exp()

# set inference parameters
confthre = 0.0001
nmsthre = cfg.nmsthre

# get YOLOX model
model = exp.get_model()
model.cuda()
model.eval()
model.head.training=False
model.training=False

# get custom trained checkpoint
ckpt_file = f"{cfg.absolute_path}/results/{config}/fold{fold}/{config}/best_ckpt.pth"
ckpt = torch.load(ckpt_file, map_location="cpu")
model.load_state_dict(ckpt["model"])
for mode, df in zip(['oof', 'test'], [val, cfg.test_df]):
    if ((mode == 'oof') & (not cfg.predict_valid) or ((mode == 'test') & (not cfg.predict_test))):
        continue
    print('inference', mode, 'len(df):', len(df.drop_duplicates('path')))
    ds = MyDataset(cfg, df)
    loader = DataLoader(ds, batch_size=cfg.batch_size*2, shuffle=False, drop_last=False,
                      num_workers=cpu_count(), worker_init_fn=worker_init_fn)

    preds = []
    all_paths = []
    all_ratios = []
    with torch.no_grad():
        for loader_n, input in enumerate(loader):
            if loader_n % 100 == 0:
                print(loader_n)
            images, paths, ratios = input
            images = images.cuda()
            outputs = model(images)
            outputs = postprocess(
                        outputs, len(categories), confthre,
                        nmsthre, class_agnostic=True
                    )
            preds += outputs
            all_paths += list(paths)
            all_ratios += list(ratios)

    dfs = []
    all_boxes = []
    all_class_ids = []
    all_scores = []
    for n, (predictions, path, ratio)  in enumerate(zip(preds, all_paths, all_ratios)):
        if predictions is None:
            continue
        predictions = predictions.cpu().numpy()

        bboxes = predictions[:, 0:4]

        bboxes /= ratio
        bboxes = bboxes.tolist()
        bbclasses = predictions[:, 6]
        scores = predictions[:, 4] * predictions[:, 5]
        path_df = df[df.path == path].iloc[:1]
        for box, score, class_id in zip(bboxes, scores, bbclasses):
            all_boxes.append(box)
            all_scores.append(score)
            all_class_ids.append(class_id)
            dfs.append(path_df)

    df = pd.concat(dfs)
    df['class_id'] = all_class_ids
    df['class_id'] = df['class_id'].astype(int)
    df['class_name'] = df['class_id'].map(class_id_name_map)
    df['conf'] = all_scores
    df[['x_min', 'y_min', 'x_max', 'y_max']] = all_boxes
    df[['x_min', 'y_min', 'x_max', 'y_max']] = np.round(df[['x_min', 'y_min', 'x_max', 'y_max']]).astype(int)
    df.to_csv(f'{cfg.absolute_path}/results/{config}/{mode}_fold{fold}.csv', index = False)
    print('save to', f'{cfg.absolute_path}/results/{config}/{mode}_fold{fold}.csv, len:', len(df))
    del df, dfs, all_boxes
    gc.collect()
print(f'command: mv {config_path} {cfg.absolute_path}/results/{args.config}/')
os.system(f'mv {config_path} {cfg.absolute_path}/results/{args.config}/')
