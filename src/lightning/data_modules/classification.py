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

class ClassificationDataset(Dataset):
    def __init__(self, df, transforms, cfg, phase, current_epoch=None):
        self.transforms = transforms
        self.paths = df.path.values
        # self.paths = df.origin_path.values
        self.cfg = cfg
        self.phase = phase
        self.current_epoch = current_epoch
        if phase != 'test':
            self.labels = df[cfg.label_features].values
        if (self.cfg.box_crop is not None) and (self.cfg.box_crop):
            self.boxes = df[['x_min', 'y_min', 'x_max', 'y_max']].astype(int).values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = cv2.imread(path)[:,:,::-1]  # [:,:,::-1] 對圖片進行反轉

        if self.cfg.box_crop:
            box = self.boxes[idx]
            x_pad = (box[2] - box[0])//2 * self.cfg.box_crop_x_ratio
            y_pad = (box[3] - box[1])//2 * self.cfg.box_crop_y_ratio
            x_min = np.max([box[0]-x_pad, 0])
            y_min = np.max([box[1]-y_pad, 0])
            if hasattr(self.cfg, 'box_crop_y_upper_ratio'):  # 如果 cfg 中定義了 box_crop_y_upper_ratio
                y_upper_pad = (box[3] - box[1])//2 * self.cfg.box_crop_y_upper_ratio
                y_min = np.max([box[1]-y_upper_pad, 0])
            x_max = np.min([box[2]+x_pad, image.shape[1]])
            y_max = np.min([box[3]+y_pad, image.shape[0]])
            s = image.shape
            image = image[int(y_min):int(y_max), int(x_min):int(x_max), :]  # : 表示所有通道 (例如 RGB 的 3 個通道)

        if self.transforms:
            image = self.transforms(image=image)['image']

        if self.phase == 'test':
            return image

        label = self.labels[idx]

        return image, torch.FloatTensor(label)

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

def angle_of_line(x1, y1, x2, y2):
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
        if ((hasattr(cfg, 'windowing_3ch_v1')) and (cfg.windowing_3ch_v1)):
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
        self.phase = phase
        self.current_epoch = current_epoch
        self.l_points = df[['l_x', 'l_y']].values
        self.r_points = df[['r_x', 'r_y']].values
        if phase != 'test':
            self.labels = df[cfg.label_features].values

    def __len__(self):
        return len(self.paths)

    def load_image(self, path, a, b, origin_size):
        if path == 'nan':
            image = np.zeros((origin_size[0], origin_size[1], 3)).astype(np.uint8)
        else:
            try:
                image = np.load(path)
                image = cv2.resize(image, (origin_size[1], origin_size[0]))
                image = np.concatenate([image, image[:,:,[0]]], 2)
            except:
                image = cv2.imread(path)
                try:
                    image = cv2.resize(image, (origin_size[1], origin_size[0]))
                except:
                    image = np.zeros((origin_size[0], origin_size[1], 3)).astype(np.uint8)

        rotate_angle = angle_of_line(a[0], a[1], b[0], b[1])
        transform = A.Compose([
            A.Rotate(limit=(-rotate_angle, -rotate_angle), p=1.0),
        ], keypoint_params= A.KeypointParams(format='xy', remove_invisible=False),
        )

        t = transform(image=image, keypoints=[a, b])
        image = t["image"]
        a, b = t["keypoints"]
        if a[0]<0:
            a = (0, a[1])
        if b[0]<0:
            b = (0, b[1])
        if a[1]<0:
            a = (a[0], 0)
        if b[1]<0:
            b = (b[0], 0) 
        if self.cfg.box_crop:
            if ((hasattr(self.cfg, 'xy_center_point')) and (self.cfg.xy_center_point)):
                x = int((a[0]+b[0])/2)
                y = int((a[1]+b[1])/2)
            else:
                x = int(b[0])
                y = int(b[1])
            
            w = abs(b[0]-a[0])
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
            origin_size = cv2.imread(path).shape[:2]

        study_id = self.study_ids[idx]
        a = self.l_points[idx]
        b = self.r_points[idx]

        images = []
        for path in paths.split(','):
            image = self.load_image(path, a, b, origin_size)

            try:
                images.append(self.transforms(image=image.astype(np.uint8))['image'])
            except:
                print(path)
                raise

        images = np.stack(images, 0)
        images = torch.tensor(images).float()

        if self.phase == 'train' and random.random() < self.cfg.p_rand_order_v1:
            indices = torch.randperm(images.size(0))
            images = images[indices]

        if self.phase == 'test':
            return images

        label = self.labels[idx]

        return images, torch.FloatTensor(label)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
def get_dataset_class(cfg):
    if ((hasattr(cfg, 'use_sagittal_mil_dataset')) and (cfg.use_sagittal_mil_dataset)):
        claz = SagittalMILDataset
    else:
        claz = ClassificationDataset
    return claz

def my_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    return images, labels

def my_collate_fn(batch):
    images = [item[0] for item in batch]
    images = torch.stack(images, dim=0)

    if isinstance(batch[0][1], tuple):
        labels = [item[1][0] for item in batch]
        masks = [item[1][1] for item in batch]
        labels = torch.stack(labels, dim=0)
        masks = torch.stack(masks, dim=0)
        return images, (labels, masks)
    else:
        labels = [item[1] for item in batch]
        labels = torch.stack(labels, dim=0)
        return images, labels

class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage):
        pass

    def train_dataloader(self):
        if self.cfg.train_by_all_data:
            tr = self.cfg.train_df
        else:
            tr = self.cfg.train_df[self.cfg.train_df.fold != self.cfg.fold]
        self.tr = tr
        if self.cfg.upsample is not None:
            assert type(self.cfg.upsample) == int
            origin_len = len(tr)
            dfs = [tr]
            for col in self.cfg.label_features:
                for _ in range(self.cfg.upsample):
                    dfs.append(tr[tr[col]==1])
            tr = pd.concat(dfs)
            print(f'upsample, len: {origin_len} -> {len(tr)}')

        print('len(train):', len(tr))
        claz = get_dataset_class(self.cfg)
        if getattr(self.cfg, 'use_custom_sampler', False):
            tr = tr.reset_index(drop=True)
        train_ds = claz(
            df=tr,
            transforms=self.cfg.transform['train'],
            cfg=self.cfg,
            phase='train',
            current_epoch=self.trainer.current_epoch,
        )
        if getattr(self.cfg, 'use_custom_sampler', False):
            return DataLoader(train_ds, batch_size=self.cfg.batch_size,
                num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn,
                sampler=InterleavedMaskClassBatchSampler(tr, self.cfg))

        else:
            return DataLoader(train_ds, batch_size=self.cfg.batch_size, pin_memory=True, shuffle=True, drop_last=True,
                num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn, collate_fn=my_collate_fn)

    def val_dataloader(self):
        val = get_val(self.cfg)
        self.val = val

        print('len(valid):', len(val))
        claz = get_dataset_class(self.cfg)

        valid_ds = claz(
            df=val,
            transforms=self.cfg.transform['val'],
            cfg=self.cfg,
            phase='valid'
        )

        return DataLoader(valid_ds, batch_size=self.cfg.batch_size, pin_memory=True, shuffle=False, drop_last=False,
                          num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn, collate_fn=my_collate_fn)