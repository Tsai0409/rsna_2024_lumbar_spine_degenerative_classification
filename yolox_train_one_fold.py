# yolox_train_one_fold.py
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import json
import pandas as pd
import numpy as np
import cv2
import gc
from pdb import set_trace as st
from tqdm.notebook import tqdm
tqdm.pandas()
from sklearn.model_selection import GroupKFold
from src.yolo_configs import *

class NumpyEncoder(json.JSONEncoder):  # json.JSONEncoder 的自定義編碼器 NumpyEncoder，用來處理 NumPy 特有的資料類型。這樣可以確保在將 NumPy 物件轉換為 JSON 格式時不會出現錯誤
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# save_annot_json(train_annot_json, f"{cfg.absolute_path}/input/annotations/{train_json_filename}")
# filename = '/kaggle/working/duplicate/input/annotations/train_rsna_axial_all_images_left_yolox_x___train_axial_for_yolo_all_image_v1_fold0_len9602.json'
def save_annot_json(json_annotation, filename):  # filename 是 json 路徑
    json.dump(json_annotation, open(filename, 'w'), indent=4, cls=NumpyEncoder)  # json.dump() 是 Python 的 json 模組中用來將 Python 物件寫入 JSON 檔案的函數；open(filename, 'w') 打開指定的檔案（這裡是 filename）以進行寫入模式

annotion_id = 0
image_id_n = 0
def dataset2coco(df):  # COCO 是一種常用的物件檢測資料格式，包含了圖片、標註(bounding boxes）、類別等資訊；從資料框 df 中提取資料，並格式化為 COCO 所需的結構
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
    # annotations_json["info"].append(info)
    annotations_json["info"] = info

    lic = {
            "id": 1,
            "url": "",
            "name": "Unknown"
        }
    annotations_json["licenses"].append(lic)

    for id_n, (path, idf) in enumerate(df.groupby('path')):  # enumerate() 會返回一個可迭代的對象，每次迭代會返回一個元組 (index, value)；以一張 image 為單位(group)
        images = {
            "id": image_id_n,  # 每處理一張圖片時，image_id_n 的值會自動 +1
            "license": 1,
            "file_name": path,  # /kaggle/temp/axial_all_images/2767326159___223384___5.png
            "height": idf.image_height.values[0],
            "width": idf.image_width.values[0],
            "date_captured": "2023-04-10T15:01:26+00:00"
        }
        annotations_json["images"].append(images)
        
        for _, row in idf.iterrows():  # iterrows() 會返回每一行的索引(用 _ 忽略這個索引)和該行的資料(row)
            # 如果每張圖片只有一個標註資料，那麼 for _, row in idf.iterrows(): 會執行一次
            # 如果每張圖片有多個標註資料，則 for _, row in idf.iterrows(): 會執行多次，每次處理一個標註資料
            # Axial T2 -> Subarticular Stenosis(有左右邊的(x, y) 的標註點，而 path 有可能一樣) 
            bbox = row[['x_min', 'y_min', 'x_max', 'y_max']].values
            b_width = bbox[2]-bbox[0]
            b_height = bbox[3]-bbox[1]

            image_annotations = {
                "id": annotion_id,
                "image_id": image_id_n,
                "category_id": row.class_id,  # class_id = [0, 1, 2, 3, 4]
                "bbox": [bbox[0], bbox[1], b_width, b_height],  # bbox: [x_min, y_min, b_width, b_height],
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
parser.add_argument("--config", '-c', type=str, default='Test', help="config name in yolo_configs.py")  # configs=("rsna_axial_all_images_left_yolox_x" "rsna_axial_all_images_right_yolox_x")
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

# DATA_KAGGLE_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"
# self.train_df = f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_6/train_axial_for_yolo_all_image_v1.csv'
# self.test_df = f'{WORKING_DIR}/csv_train/preprocess_4/train_with_fold.csv'

# absolute_path = /kaggle/working/duplicate
print('absolute_path = '+cfg.absolute_path)
# cfg.train_df.path = cfg.absolute_path + '/' + cfg.train_df.path  # train 照片路徑；ex:/kaggle/temp/axial_all_images/2767326159___223384___5.png
cfg.train_df.path = cfg.train_df.path
# cfg.test_df.path = cfg.absolute_path + '/' + cfg.test_df.path
cfg.test_df.path = cfg.test_df.path

cfg.train_df.class_id = cfg.train_df.class_id.astype(int)  # 確保為 int 型態
# 這段應該不會執行到：
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
os.chdir('/kaggle/working/duplicate/src/YOLOX')  # os.chdir() 是一個 Python 函式，用來改變當前的工作目錄
# sys.path.append() 並不會改變當前工作目錄，僅僅是告訴 Python 去某個目錄尋找模組。這對檔案操作不會有影響
# os.chdir() 會更改當前的工作目錄，這會影響檔案操作，但它不會改變 Python 模組的搜尋路徑；差異？

print(f'\n----------------------- Config -----------------------')
config_str = ''
for k, v in vars(cfg).items():
    if (k == 'model') | ('df' in k):
        continue
    print(f'\t{k}: {v}')
    config_str += f'{k}: {v}, '
print(f'----------------------- Config -----------------------\n')

# absolute_path = /kaggle/working/duplicate
# configs=("rsna_axial_all_images_left_yolox_x" "rsna_axial_all_images_right_yolox_x")
config_path = f'configfile_{config}_fold{fold}.py'  # 創立新的配置檔案 config_path = configfile_rsna_axial_all_images_left_yolox_x_fold0.py
os.makedirs(f'{cfg.absolute_path}/results/{config}', exist_ok=True)  # 創立一個目錄 /kaggle/working/duplicate/results/rsna_axial_all_images_left_yolox_x

# self.model_name_for_yolox 沒有出現(all condition)
if hasattr(cfg, 'model_name_for_yolox'):
    model_name = cfg.model_name_for_yolox
else:  # here
    # self.pretrained_path = '/kaggle/input/pretrain-7/yolox_x.pth'
    model_name = cfg.pretrained_path.split('/')[-1].replace('.pth', '')  # model_name = yolox

categories = []
class_id_name_map = {}

# class_id = [0, 1, 2, 3, 4] -- wrong
# class_name = [L1/L2, L2/L3, L3/L4, L4/L5, L5/S1] -- wrong
# class_id = [0, 0] 但我不知道為什麼是這樣 -> 在 yolo_configs 中定義 class_id 及 class_name
# class_name = [left, right]
for n, (c, id) in enumerate(zip(cfg.train_df.sort_values('class_id').class_name.unique(), cfg.train_df.sort_values('class_id').class_id.unique())):
    classes = {'supercategory': 'none'}  # 創建了一個名為 classes 的字典
    classes['id'] = id  # 以 (key, value) pair 的形式存放
    classes['name'] = c  # 以 (key, value) pair 的形式存放
    categories.append(classes)  # 將 classes 的字典存到 catagories 的 list 中；這邊有 5 個 class_id 所以有 5 個字典？
    class_id_name_map[id] = c
print('class_id_name_map:', class_id_name_map)  # class_id_name_map: {0: 'left'} -> yolo.configs

tr = cfg.train_df[cfg.train_df.fold != fold]  # DataFrame
val = cfg.train_df[cfg.train_df.fold == fold]
print('len(train) / len(val):', len(tr), len(val))
# class_id_name_map: {0: 'left'} -> len(train) / len(val): 9602 1924
# class_id_name_map: {0: 'right'} -> len(train) / len(val): 9611 1924

# self.train_df_path = f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_6/train_axial_for_yolo_all_image_v1.csv'
train_df_filename = args.config + '___' + cfg.train_df_path.split('/')[-1].replace('.csv', '')  # rsna_axial_all_images_left_yolox_x___train_axial_for_yolo_all_image_v1
train_json_filename = f'train_{train_df_filename}_fold{fold}_len{len(tr)}.json'  # train_rsna_axial_all_images_left_yolox_x___train_axial_for_yolo_all_image_v1_fold0_len9602.json
valid_json_filename = f'valid_{train_df_filename}_fold{fold}_len{len(val)}.json'  # vaild_rsna_axial_all_images_left_yolox_x___train_axial_for_yolo_all_image_v1_fold0_len1924.json

# class rsna_axial_all_images_left_yolox_x、class rsna_axial_all_images_right_yolox_x -> cfg.inference_only=False
# class rsna_10classes_yolox_x -> cfg.inference_only=True
if not cfg.inference_only:  # configs=("rsna_axial_all_images_left_yolox_x" "rsna_axial_all_images_right_yolox_x")
    # self.update_json = False (all condition)
    if os.path.exists(f"{cfg.absolute_path}/input/annotations/{train_json_filename}") & os.path.exists(f"{cfg.absolute_path}/input/annotations/{valid_json_filename}") & (not cfg.update_json):
        print('make labels skip.')
    else:
        print('make labels start...')  # here
        train_annot_json = dataset2coco(tr)  # 會用到 cfg.train_df.path = cfg.absolute_path + '/' + cfg.train_df.path；可能有錯 -> 已修正
        valid_annot_json = dataset2coco(val)
        os.system(f'mkdir -p {cfg.absolute_path}/input/annotations/')  # 創建 annotations 的目錄；/kaggle/working/duplicate/input/annotations/
        save_annot_json(train_annot_json, f"{cfg.absolute_path}/input/annotations/{train_json_filename}")  # 把 def dataset2coco(df): 的資料存到 json 裡面
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
        # self.model_name= 'yolov5m' (all confdition)
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
        self.max_epoch = {cfg.epochs}  # self.epochs = 20 (original 40)
        self.train_ann = "{cfg.absolute_path}/input/annotations/{train_json_filename}"  # self.train_ann = '/kaggle/working/duplicate/input/annotations/train_rsna_axial_all_images_left_yolox_x___train_axial_for_yolo_all_image_v1_fold0_len9602.json'
        self.val_ann = "{cfg.absolute_path}/input/annotations/{valid_json_filename}"
        self.output_dir = "{cfg.absolute_path}/results/{config}/fold{fold}"  # absolute_path = '/kaggle/working/duplicate/results/train_rsna_axial_all_images_left_yolox_x/fold0'
        self.input_size = {cfg.image_size}  # self.image_size = (512, 512)(all condition)
        self.test_size = {cfg.image_size}
        self.no_aug_epochs = {cfg.no_aug_epochs}  # self.no_aug_epochs = 15
        self.warmup_epochs = {cfg.warmup_epochs}  # self.warmup_epochs = 5
        self.num_classes = {cfg.train_df.class_name.nunique()}  # class_name = [L1/L2, L2/L3, L3/L4, L4/L5, L5/S1];self.num_classes = 5
        self.categories = {categories}
        self.class_id_name_map = {class_id_name_map}
        ### need change ###

        ### fyi ###
        self.data_num_workers = {cfg.batch_size}  # self.batch_size = 8 (all condition)
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
        self.nmsthre = {cfg.nmsthre}  # self.nmsthre = 0.45
        ### fyi ###

        # self.heavy_aug = False (all condition)
        if {cfg.heavy_aug}:
            self.scale = (0.1, 2)
            self.mosaic_scale = (0.8, 1.6)
            self.perspective = 0.0
'''

# config_path = configfile_rsna_axial_all_images_left_yolox_x_fold0.py
with open(config_path, 'w') as f:
    f.write(config_file_template)  # 把上面的參數寫到.py 裡面

from pycocotools.coco import COCO
from random import sample
import importlib

# class rsna_axial_all_images_left_yolox_x、class rsna_axial_all_images_right_yolox_x -> cfg.inference_only=False
# class rsna_10classes_yolox_x -> cfg.inference_only=True
if cfg.inference_only:
    print('inference_only.')
else:  # here
    print('train start...')
    # train_str = f'python train.py -f {config_path} -d 1 -b {cfg.batch_size} --fp16 -o -c {cfg.pretrained_path}'
    # train_str = f'python tools/train.py -f {config_path} -d 1 -b {cfg.batch_size} -fp16 -o -c {cfg.pretrained_path}'  # self.pretrained_path = '/kaggle/input/pretrain-7/yolox_x.pth' (all condition)
    train_str = f'python tools/train.py -f {config_path} -d 1 -b {cfg.batch_size} --fp16 -o -c {cfg.pretrained_path}'

    # self.resume = False (all condition)
    if cfg.resume:  # no here
        train_str = f'python tools/train.py -f {config_path} -d 1 -b {cfg.batch_size} --fp16 -o -c {cfg.absolute_path}/results/{config}/fold{fold}/{config}/best_ckpt.pth --resume --start_epoch {cfg.resume_start_epoch}'
        print('using cfg.resume')


    print('train_str:', train_str)
    # train_str: python tools/train.py -f configfile_rsna_axial_all_images_left_yolox_x_fold0.py -d 1 -b 8 --fp16 -o -c /kaggle/input/pretrain-7/yolox_x.pth
    os.system(train_str)

### inference ###
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count
# sys.path.append('')
sys.path.append(f'{cfg.absolute_path}/src/YOLOX')  # sys.path.append() 加入到 Python 模組的搜尋路徑中；/kaggle/working/duplicate/src/YOLOX

from yolox.utils import postprocess
from yolox.data.data_augment import ValTransform

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)  # 確保在多 worker 環境下，每個 worker 的隨機數生成器都有獨立且不同的初始狀態，從而避免各個 worker 產生相同的隨機數序列

# ds = MyDataset(cfg, val)
class MyDataset(Dataset):
    def __init__(self, cfg, df):
        self.paths = df.path.unique()
        self.cfg = cfg
        self.preproc = ValTransform(legacy = False)

    def __len__(self): 
        return len(self.paths)

    def _read_image(self, path):
        if '.npy' in path:  # .npy，代表該檔案是以 NumPy 格式存儲的圖片
            image = np.load(path)
            image = np.array([image, image, image]).transpose((1,2,0))
        else:
            image = cv2.imread(path)
        
        return image

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = self._read_image(path)
        ratio = min(cfg.image_size[0] / img.shape[0], cfg.image_size[1] / img.shape[1])  # 計算了一個縮放比例 ratio(確保縮放時不會超過目標尺寸)
        # ratio = min(512 / img.shape[0], 512 / img.shape[1])

        img, _ = self.preproc(img, None, cfg.image_size)  # 圖像預處理
        img = torch.from_numpy(img).float()  # 轉換成 PyTorch Tensor
        img = img.float()

        return img, path, ratio

# get YOLOX experiment
current_exp = importlib.import_module(config_path.replace('.py', ''))  # config_path = configfile_rsna_axial_all_images_left_yolox_x_fold0.py (紀錄訓練 model 的參數)
exp = current_exp.Exp()

# set inference parameters
confthre = 0.0001
nmsthre = cfg.nmsthre  # self.nmsthre = 0.45 (all condition)

# get YOLOX model
model = exp.get_model()
model.cuda()
model.eval()
model.head.training=False
model.training=False

# get custom trained checkpoint
ckpt_file = f"{cfg.absolute_path}/results/{config}/fold{fold}/{config}/best_ckpt.pth"  # ckpt_file = "/kaggle/working/duplicate/results/train_rsna_axial_all_images_left_yolox_x/fold0/train_rsna_axial_all_images_left_yolox_x/best_ckpt.pth"
ckpt = torch.load(ckpt_file, map_location="cpu")
model.load_state_dict(ckpt["model"])
for mode, df in zip(['oof', 'test'], [val, cfg.test_df]):  # (vaild, test) -> first (oof, val), second (test, cfg.test_df)
    # self.predict_valid = True
    if ((mode == 'oof') & (not cfg.predict_valid) or ((mode == 'test') & (not cfg.predict_test))):
        continue
    print('inference', mode, 'len(df):', len(df.drop_duplicates('path')))
    ds = MyDataset(cfg, df)  # return img, path, ratio
    loader = DataLoader(ds, batch_size=cfg.batch_size*2, shuffle=False, drop_last=False, num_workers=cpu_count(), worker_init_fn=worker_init_fn)

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
                        outputs, len(categories), confthre, nmsthre, class_agnostic=True
                    )
            preds += outputs  # output 的資料型態是什麼？(為什麼能 return 是 ['x_min', 'y_min', 'x_max', 'y_max'] 呢？)
            all_paths += list(paths)
            all_ratios += list(ratios)

    dfs = []
    all_boxes = []
    all_class_ids = []
    all_scores = []
    for n, (predictions, path, ratio)  in enumerate(zip(preds, all_paths, all_ratios)):
        if predictions is None:
            continue  # 直接進入下一次迭代
        predictions = predictions.cpu().numpy()

        bboxes = predictions[:, 0:4]  # predictions 以二維陣列的方式回傳；predictions[:, 0:4] 取出所有 row 的前四個 col

        bboxes /= ratio
        bboxes = bboxes.tolist()
        bbclasses = predictions[:, 6]  # 提取每個預測框的類別標籤
        scores = predictions[:, 4] * predictions[:, 5]  # predictions[:, 4] 代表物體存在的置信度；predictions[:, 5] 代表分類概率
        path_df = df[df.path == path].iloc[:1]  # .iloc[:1] 只取第一行
        for box, score, class_id in zip(bboxes, scores, bbclasses):
            all_boxes.append(box)
            all_scores.append(score)
            all_class_ids.append(class_id)
            dfs.append(path_df)  # dfs 是一個 Python 列表

    df = pd.concat(dfs)  # df 是一個 DataFrame
    df['class_id'] = all_class_ids
    df['class_id'] = df['class_id'].astype(int)
    df['class_name'] = df['class_id'].map(class_id_name_map)
    df['conf'] = all_scores
    df[['x_min', 'y_min', 'x_max', 'y_max']] = all_boxes
    df[['x_min', 'y_min', 'x_max', 'y_max']] = np.round(df[['x_min', 'y_min', 'x_max', 'y_max']]).astype(int)  # 經過四捨五入並轉換成整數
    df.to_csv(f'{cfg.absolute_path}/results/{config}/{mode}_fold{fold}.csv', index = False)
    print('save to', f'{cfg.absolute_path}/results/{config}/{mode}_fold{fold}.csv, len:', len(df))
    # inference oof len(df): 1924 -> save to /kaggle/working/duplicate/results/rsna_axial_all_images_left_yolox_x/oof_fold0.csv, len: 4612
    # inference test len(df): 10598 -> save to /kaggle/working/duplicate/results/rsna_axial_all_images_left_yolox_x/test_fold0.csv, len: 26727
    # inference oof len(df): 1924 -> save to /kaggle/working/duplicate/results/rsna_axial_all_images_right_yolox_x/oof_fold0.csv, len: 6003
    # inference test len(df): 10598 -> save to /kaggle/working/duplicate/results/rsna_axial_all_images_right_yolox_x/test_fold0.csv, len: 34149

    del df, dfs, all_boxes
    gc.collect()
print(f'command: mv {config_path} {cfg.absolute_path}/results/{args.config}/')
# command: mv rsna_axial_all_images_left_yolox_x /kaggle/working/duplicate/results/rsna_axial_all_images_left_yolox_x/
os.system(f'mv {config_path} {cfg.absolute_path}/results/{args.config}/')

# 如何辨別時 Left/Right 呢 -> yolo.configs 中在 training data 就分好了