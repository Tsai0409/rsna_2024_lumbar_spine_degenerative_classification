# src/lighting/data_modules/classification.py
import numpy as np
import pandas as pd
import json
from pathlib import Path
import pickle
from glob import glob
import cv2
from PIL import Image
import random
import albumentations as A

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pdb import set_trace as st

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pytorch_lightning as pl
from .util import *
from .mil import MilDataset
import os
import librosa
from multiprocessing import Pool, cpu_count
from pfio.cache import MultiprocessFileCache
from monai.transforms import Resize
from albumentations import ReplayCompose

def sigmoid(x):
    return 1/(1 + np.exp(-x))

import pdb
import sys
# ForkedPdb().set_trace()

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

class ClassificationDataset(Dataset):  # image label
    def __init__(self, df, transforms, cfg, phase, current_epoch=None):
        self.transforms = transforms
        self.paths = df.path.values  # df = train_for_sagittal_level_cl_v1_for_train_spinal_only.csv
        self.cfg = cfg
        self.phase = phase  # phase='train' (defined below)
        self.current_epoch = current_epoch
        # phase='train', 'vaild'
        if phase != 'test':
            self.labels = df[cfg.label_features].values  # 按照 label_features = ['label1', 'label2'...]
        
        # cfg.box_crop = None -> silce estimation(sagittal)
        # cfg.box_crop = Ture -> axial classificaion\sagittal classification
        if (self.cfg.box_crop is not None) and (self.cfg.box_crop):
            self.boxes = df[['x_min', 'y_min', 'x_max', 'y_max']].astype(int).values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]  # image path(including temp)
        image = cv2.imread(path)[:,:,::-1]  # [:,:,::-1] 對圖片進行反轉 (BGR -> RGB)

        # cfg.box_crop = None -> silce estimation(sagittal)
        # cfg.box_crop = Ture -> axial classificaion\sagittal classification
        if self.cfg.box_crop:
            box = self.boxes[idx]
            x_pad = (box[2] - box[0])//2 * self.cfg.box_crop_x_ratio
            y_pad = (box[3] - box[1])//2 * self.cfg.box_crop_y_ratio
            x_min = np.max([box[0]-x_pad, 0])
            y_min = np.max([box[1]-y_pad, 0])
            
            if hasattr(self.cfg, 'box_crop_y_upper_ratio'):  # 如果 cfg 中定義了 box_crop_y_upper_ratio；box_crop_y_upper_ratio 沒有使用
                y_upper_pad = (box[3] - box[1])//2 * self.cfg.box_crop_y_upper_ratio
                y_min = np.max([box[1]-y_upper_pad, 0])

            x_max = np.min([box[2]+x_pad, image.shape[1]])
            y_max = np.min([box[3]+y_pad, image.shape[0]])
            s = image.shape
            image = image[int(y_min):int(y_max), int(x_min):int(x_max), :]  # : 表示所有通道 (例如 RGB 的 3 個通道)

        # self.transforms = True -> transforms=self.cfg.transform['train']
        if self.transforms:  # here
            image = self.transforms(image=image)['image']  
            # 使用 Albumentations(A) 的轉換管道時，輸出結果中的 image 是一個 numpy array(numpy.ndarray)，其形狀通常是(height, width, channels)
            # 最後經過 ToTensorV2()，輸出的 image 就會是 torch.Tensor

        if self.phase == 'test':  # 如果 test 的話，會傳 image 而已，沒有 label
            return image

        label = self.labels[idx]
        return image, torch.FloatTensor(label)  # 看起來是拿 全部的圖片 找對應的 ['label1', 'label2'...] 作為 model 的 input

import math

def crop_between_keypoints(img, keypoint1, keypoint2, ratio=0.1):
    h, w = img.shape[:2]
    x1, y1 = int(keypoint1[0]), int(keypoint1[1])
    x2, y2 = int(keypoint2[0]), int(keypoint2[1])
    
    # Calculate bounding box around the keypoints
    left = int(min(x1, x2))
    right = int(max(x1, x2))
    top = int(min(y1, y2) - (h * 0.1))
    bottom = int(max(y1, y2) + (h * 0.1))
            
    # Crop the image
    return img[top:bottom, left:right, :]

def angle_of_line(x1, y1, x2, y2):  # 「這條線要旋轉多少度，才能變水平」的角度
    return math.degrees(math.atan2(-(y2-y1), x2-x1))

import random
import time
import numpy as np
from torch.utils.data import Sampler

class InterleavedMaskClassBatchSampler(Sampler):
    def __init__(self, df, cfg):
        self.df = df
        self.batch_size = cfg.batch_size
        self.indices_by_class = {cls: list(df[df['mask_class'] == cls].index) for cls in df['mask_class'].unique()}
        for indices in self.indices_by_class.values():
            np.random.shuffle(indices)

    def __iter__(self):
        all_classes = list(self.indices_by_class.keys())
        while len(all_classes) > 0:
            cls = np.random.choice(all_classes)
            if len(self.indices_by_class[cls]) >= self.batch_size:
                print(self.indices_by_class[cls][:self.batch_size])
                for _ in range(self.batch_size):
                    yield self.indices_by_class[cls].pop()
            else:
                all_classes.remove(cls)

    def __len__(self):
        return sum(len(indices) for indices in self.indices_by_class.values())

class InterleavedMaskClassBatchSampler(Sampler):
    def __init__(self, df, cfg):
        df['tmp_for_batch_sampler'] = list(range(len(df)))
        self.batch_size = cfg.batch_size
        self.df = df
        self.init_indices()
        self.total_len = len(self.batch_indices_list) * self.batch_size
        # st()
        # df[df.tmp_for_batch_sampler.isin(self.batch_indices_list[1])]

    def init_indices(self):
        chunks = []
        for c in [0, 1, 34, 4]:
            cdf = self.df[self.df.mask_class == c]
            cdf = cdf.sample(len(cdf))
            lst = cdf.tmp_for_batch_sampler.values.tolist()
            chunks += [lst[i:i+self.batch_size] for i in range(0, len(lst), self.batch_size) if len(lst[i:i+self.batch_size]) == self.batch_size]

        self.batch_indices_list = random.sample(chunks, len(chunks))

    def __iter__(self):
        for batch_indices in self.batch_indices_list:
            for idx in batch_indices:
                yield idx

    def __len__(self):
        return self.total_len

class InterleavedMaskClassBatchSamplerBK(Sampler):
    def __init__(self, df, cfg):
        self.df = df
        self.batch_size = cfg.batch_size
        self.init_indices()

    def init_indices(self):
        self.indices_by_class = {cls: list(self.df[self.df['mask_class'] == cls].index) for cls in self.df['mask_class'].unique()}
        for indices in self.indices_by_class.values():
            np.random.shuffle(indices)

    def __iter__(self):
        self.init_indices()  # 各エポックの開始時にインデックスを再初期化
        all_classes = list(self.indices_by_class.keys())
        while len(all_classes) > 0:
            cls = np.random.choice(all_classes)
            while len(self.indices_by_class[cls]) >= self.batch_size:
                for _ in range(self.batch_size):
                    yield self.indices_by_class[cls].pop()
            if len(self.indices_by_class[cls]) < self.batch_size and len(self.indices_by_class[cls]) > 0:
                for _ in range(len(self.indices_by_class[cls])):
                    yield self.indices_by_class[cls].pop()
            all_classes.remove(cls)

    def __len__(self):
        return sum(len(indices) for indices in self.indices_by_class.values())

def collate_fn(batch):
    images, targets= list(zip(*batch))
    # images = torch.stack(images)
    # targets = torch.stack(targets)
    return images, targets

class SagittalMILDataset(Dataset):
    def __init__(self, df, transforms, cfg, phase, current_epoch=None):
        self.transforms = transforms
        self.paths = df.path.values
        if ((hasattr(cfg, 'windowing_3ch_v1')) and (cfg.windowing_3ch_v1)):  # windowing_3ch_v1 沒有出現
            self.windowing_paths = df.windowing_path.values
        self.paths_list = df.paths.values
        self.study_ids = df.study_id.values
        for path in self.paths:
            if not os.path.exists(path):
                print('-'*1000)
                print(f'error!! not exists {path}')
                print('-'*1000)
                raise
        self.cfg = cfg
        self.phase = phase  # train -> test
        self.current_epoch = current_epoch
        self.l_points = df[['l_x', 'l_y']].values
        self.r_points = df[['r_x', 'r_y']].values
        if phase != 'test':  # train -> test
            self.labels = df[cfg.label_features].values

    def __len__(self):
        return len(self.paths)

    def load_image(self, path, a, b, origin_size):  # path(conf 最高的 slice)、a = ['l_x', 'l_y']、b = ['r_x', 'r_y']
        if path == 'nan':
            image = np.zeros((origin_size[0], origin_size[1], 3)).astype(np.uint8)
        else:
            try:
                image = np.load(path)
                image = cv2.resize(image, (origin_size[1], origin_size[0]))
                image = np.concatenate([image, image[:,:,[0]]], 2)
            except:  # here
                image = cv2.imread(path)
                try:
                    image = cv2.resize(image, (origin_size[1], origin_size[0]))
                except:
                    image = np.zeros((origin_size[0], origin_size[1], 3)).astype(np.uint8)  # 當讀不到合法影像（或路徑錯誤）時，建立一張「全黑」的佔位圖

        rotate_angle = angle_of_line(a[0], a[1], b[0], b[1])  # 將兩點連線 轉為水平
        transform = A.Compose([
            A.Rotate(limit=(-rotate_angle, -rotate_angle), p=1.0),
        ], keypoint_params= A.KeypointParams(format='xy', remove_invisible=False),  # 不只要旋轉影像，還要把「左右兩個標記點」一起轉到正確的位置
        )

        t = transform(image=image, keypoints=[a, b])
        image = t["image"]
        a, b = t["keypoints"]

        if a[0]<0:  # 「把任何小於 0 的座標調整回 0」，因為負座標沒有意義
            a = (0, a[1])
        if b[0]<0:
            b = (0, b[1])
        if a[1]<0:
            a = (a[0], 0)
        if b[1]<0:
            b = (b[0], 0) 

        # box_crop = True -> sagittal classification
        if self.cfg.box_crop:
            # xy_center_point 沒有出現
            if ((hasattr(self.cfg, 'xy_center_point')) and (self.cfg.xy_center_point)):  # 中點裁切
                x = int((a[0]+b[0])/2)
                y = int((a[1]+b[1])/2)
            else:  # here；以「右側那個點 b」作為方框的錨點；單點裁切
                x = int(b[0])
                y = int(b[1])
            
            w = abs(b[0]-a[0])  # 計算 x軸 差距
            h = image.shape[0]*0.2

            crop_x = int(w * self.cfg.box_crop_x_ratio)
            crop_y = int(h * self.cfg.box_crop_y_ratio)
            x_min = max(x-crop_x, 0)
            y_min = max(y-crop_y, 0)

            image = image[y_min:y+crop_y, x_min:x+crop_x]
        else:
            image = crop_between_keypoints(image, a, b)
        return image

    def __getitem__(self, idx):
        path = self.paths[idx]
        paths = self.paths_list[idx]
        try:
            origin_size = np.load(path).shape[:2]
        except:
            origin_size = cv2.imread(path).shape[:2]  # here 對每一個 series_id 皆找一次 image_size

        study_id = self.study_ids[idx]
        a = self.l_points[idx]  # self.l_points = df[['l_x', 'l_y']].values
        b = self.r_points[idx]  # self.r_points = df[['r_x', 'r_y']].values

        images = []
        for path in paths.split(','):
            image = self.load_image(path, a, b, origin_size)

            try:
                images.append(self.transforms(image=image.astype(np.uint8))['image'])  # 把 image（此時是 NumPy array，可能還是浮點型或其他型別）轉成 uint8(整數 0–255)\
            except:
                print(path)
                raise

        images = np.stack(images, 0)  # (C, H, W) -> (N, C, H, W)，其中 N = len(images) = 5
        images = torch.tensor(images).float()

        # self.p_rand_order_v1 = 0；在「訓練階段」隨機打亂（shuffle）同一筆 sample 底下多張 slice 的順序，以做時間序列上的資料增強，這邊不這樣執行
        if self.phase == 'train' and random.random() < self.cfg.p_rand_order_v1:  
            indices = torch.randperm(images.size(0))
            images = images[indices]

        if self.phase == 'test':
            return images

        label = self.labels[idx]
        return images, torch.FloatTensor(label)

def worker_init_fn(worker_id):  # worker_id 是由 DataLoader 在啟動每個子進程時自動提供的
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
def get_dataset_class(cfg):
    # cfg.use_sagittal_mil_dataset 在crop之前沒有出現
    # cfg.use_sagittal_mil_dataset = Ture -> sagittal classification 時使用
    if ((hasattr(cfg, 'use_sagittal_mil_dataset')) and (cfg.use_sagittal_mil_dataset)):  # sagittal classification 時才會用到
        claz = SagittalMILDataset  # here -> sagittal classification
    else:
        claz = ClassificationDataset  # here -> slice estimation、axial classification
    return claz

def my_collate_fn(batch):  # 不執行
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    return images, labels

def my_collate_fn(batch):  # 遇到名稱相同的執行後者；拿 class ClassificationDataset 的輸出結果 return image, torch.FloatTensor(label) 作為 batch
    images = [item[0] for item in batch]  # (image, label)
    # 看資料的格式是 (image, label) 或 (image, (label, mask))->tuple 
    # item[0]	image
    # item[1]	(label, mask)
    # item[1][0]	label
    # item[1][1]	mask
    images = torch.stack(images, dim=0)  # 把所有影像堆疊成一個 tensor，產生的張量形狀一般為(batch_size, channels, height, width)

    if isinstance(batch[0][1], tuple):  # 檢查第一筆資料的第二個元素(即 label)是否為 tuple；如果是 tuple，代表每筆資料的 label 內包含多個元素(例如可能包含標籤和 mask)
        labels = [item[1][0] for item in batch]
        masks = [item[1][1] for item in batch]
        labels = torch.stack(labels, dim=0)
        masks = torch.stack(masks, dim=0)
        return images, (labels, masks)
    else:
        labels = [item[1] for item in batch]
        labels = torch.stack(labels, dim=0)
        return images, labels

class MyDataModule(pl.LightningDataModule):  # 我有需要知道 pl.LightningDataModule 裡面的內容嗎？
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage):
        pass

    def train_dataloader(self):
        # self.train_by_all_data = False (all condition)
        print(f"len(train_df) before filtering: {len(self.cfg.train_df)}")  # 我加
        if self.cfg.train_by_all_data:
            tr = self.cfg.train_df  # tr train input
        else:  # here
            tr = self.cfg.train_df[self.cfg.train_df.fold != self.cfg.fold]  # 選擇 train_for_sagittal_level_cl_v1_for_train_spinal_only.csv 除了當前 fold 作為訓練資料
            # tr = self.cfg.train_df[self.cfg.train_df.fold == self.cfg.fold]  # 在 sagittal_classification 中時使用；讓 train 跟 valid 的 data 是一樣的 -> 我修正的，因為只用一個 fold 訓練(同時又要作為 train 與 vaild 的資料)
            # 假設如果是完整的 5fold 還會遇到 training data 資料為空的情況嗎？
        self.tr = tr  # 現在 self.cfg.fold 的為 0，但是 self.cfg.train_df.fold 沒有 fold 以外的資料 (sagittal_spinal_range2_rolling5.csv)
        print(f"len(train_df) after filtering: {len(tr)}")  # 我加
        print(f"Current fold: {self.cfg.fold}")  # 我加


        # cfg.upsample = None
        # cfg.upsample 通常會是一個數字(1-5)，如果為1，來表示對「少數類別」的每個樣本進行一次上採樣，即複製一次(將少數類別樣本的數量變成兩倍)
        if self.cfg.upsample is not None:  # 數據集上採樣(upsampling)
            assert type(self.cfg.upsample) == int
            origin_len = len(tr)
            dfs = [tr]
            for col in self.cfg.label_features:  # 按照 label_features = ['label1', 'label2'...] 依序進行上採樣
                for _ in range(self.cfg.upsample):
                    dfs.append(tr[tr[col]==1])  # 在第一個迭代時，選擇 label1 值為 1 的數據(通常表示1的即為數據不平衡的)
            tr = pd.concat(dfs)
            print(f'upsample, len: {origin_len} -> {len(tr)}')

        print('len(train):', len(tr))
        claz = get_dataset_class(self.cfg)  # 產生 model 的 input 資料 (image, label)

        # cfg.use_custom_sampler 沒有出現在 configs -> 預設為 False
        if getattr(self.cfg, 'use_custom_sampler', False):
            tr = tr.reset_index(drop=True)
        
        train_ds = claz(  # 這邊的閱讀順序是先定義要哪一個類別 claz = get_dataset_class(self.cfg)，然後在傳如參數 -> 對
            df=tr,
            transforms=self.cfg.transform['train'],  # self.transform = medical_v3(configs);定義在：src/utils/augmentations/augmentation.py
            cfg=self.cfg,
            phase='train',
            current_epoch=self.trainer.current_epoch,  # self.trainer.current_epoch 要去哪邊找 -> src/pytorch-lightning/pytorch_lightning/trainer/trainer.py
        )

        # cfg.use_custom_sampler 沒有出現在 configs
        if getattr(self.cfg, 'use_custom_sampler', False):  # get attribute 看看是否有這個屬性；預設為False
            return DataLoader(train_ds, batch_size=self.cfg.batch_size,
                num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn,
                sampler=InterleavedMaskClassBatchSampler(tr, self.cfg))
        else:  # here
            return DataLoader(train_ds, batch_size=self.cfg.batch_size, pin_memory=True, shuffle=True, drop_last=True,
                num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn, collate_fn=my_collate_fn)

    def val_dataloader(self):
        val = get_val(self.cfg)  # get_val 可以在 src/lightning/data_modules/util.py 中找到 (below)
        self.val = val

        print('len(valid):', len(val))
        claz = get_dataset_class(self.cfg)

        valid_ds = claz(
            df=val,
            transforms=self.cfg.transform['val'],  # self.transform = medical_v3
            cfg=self.cfg,
            phase='valid'
        )

        return DataLoader(valid_ds, batch_size=self.cfg.batch_size, pin_memory=True, shuffle=False, drop_last=False,
                          num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn, collate_fn=my_collate_fn)
        # 總共有 5 fold，假設現在 fold=0，會以 fold0-4 作為 train data，而 fold0 作為 vaild data 嗎 ->
        # (Train Data) 會取 fold1 到 fold4(不包含 fold0 的部分) 被用作訓練集；前提是如果沒有啟用 train_by_all_data（也就是預設情況下）
        # (Valid Data) 會取出 cfg.train_df 中 fold0 的部分作爲驗證集
         
'''
 # cfg.valid_df = None
def get_val(cfg):
    if cfg.valid_df is None: 
        val = cfg.train_df[cfg.train_df.fold == cfg.fold] # 假設現在 fold=0 -> fold0 就是驗證資料
    else:
        val = cfg.valid_df[cfg.valid_df.fold == cfg.fold]
    return val
    # cfg.train_df.fold 這個 fold 是指在 train_df 中的 fold 欄位
'''
'''
def medical_v3(size):
    return {
        'train': A.Compose([
             A.Resize(size[0], size[1]),
             # A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5),
             A.ShiftScaleRotate(p=0.5),
             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
             A.CLAHE(clip_limit=(1,4), p=0.5),
             A.OneOf([
                 A.OpticalDistortion(distort_limit=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=1.),
                 A.ElasticTransform(alpha=3),
             ], p=0.2),
             A.OneOf([
                 A.GaussNoise(var_limit=[10, 50]),
                 A.GaussianBlur(),
                 A.MotionBlur(),
                 A.MedianBlur(),
             ], p=0.2),
             A.PiecewiseAffine(p=0.2),
             A.Sharpen(p=0.2),
             # A.Cutout(max_h_size=int(size[0] * 0.1), max_w_size=int(size[1] * 0.1), num_holes=5, p=0.5),
             A.CoarseDropout(max_height=int(size[0]*0.1), max_width=int(size[1]*0.1), max_holes=5, p=0.5),
             A.Normalize(),
             ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(),
            ToTensorV2(),
        ]),
    }
'''