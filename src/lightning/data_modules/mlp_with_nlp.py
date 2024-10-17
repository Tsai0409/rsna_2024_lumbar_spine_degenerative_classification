import numpy as np
import pandas as pd
import json
from pathlib import Path
import pickle
from glob import glob
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from pdb import set_trace as st

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .util import *
import os
from transformers import AutoTokenizer

from multiprocessing import Pool, cpu_count
# resize_transform = A.Compose([A.Resize(height=self.cfg.image_size[0], width=self.cfg.image_size[1], p=1.0)])
def load_image(args):
    path, imsize = args
    image = cv2.imread(path)[:,:,::-1]
    # 画像を拡大する場合は、 INTER_LINEARまたはINTER_CUBIC補間を使用することをお勧めします。画像を縮小する場合は、 INTER_AREA補間を使用することをお勧めします。
    # キュービック補間は計算が複雑であるため、線形補間よりも低速です。ただし、結果の画像の品質は高くなります。
    return path, cv2.resize(image, imsize, interpolation=cv2.INTER_AREA)

def pad_to_square(a, wh_ratio=4):
    if len(a.shape) == 2:
        a = np.array([a,a,a]).transpose(1,2,0)
        grayscale = True
    else:
        grayscale = False

    """ Pad an array `a` evenly until it is a square """
    if a.shape[1]>a.shape[0]*wh_ratio: # pad height
        n_to_add = a.shape[1]/wh_ratio-a.shape[0]

        pad = int(n_to_add//2)
        # bottom_pad = int(n_to_add-top_pad)
        a = np.pad(a, [(pad, pad), (0, 0), (0, 0)], mode='constant')

    elif a.shape[0]*wh_ratio>a.shape[1]: # pad width
        n_to_add = a.shape[0]*wh_ratio-a.shape[1]
        pad = int(n_to_add//2)
        # right_pad = int(n_to_add-left_pad)
        a = np.pad(a, [(0, 0), (pad, pad), (0, 0)], mode='constant')
    if grayscale:
        a = a[:,:,0]
    return a

class MlpWithNlpDataset(Dataset):
    def __init__(self, df, cfg, phase):
        self.df = df
        self.cfg = cfg
        self.phase = phase
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        text = row.text
        num_vals = row[self.cfg.num_features]
        if self.phase!='test':
            label = row.target
        else:
            label = index

        inputs = self.tokenizer.encode_plus(
                        text,
                        truncation=True,
                        add_special_tokens=True,
                        max_length=self.cfg.max_length,
                        padding='max_length',
                        return_tensors="pt"
                    )
            
        return {
            'ids': inputs['input_ids'],
            'mask': inputs['attention_mask'],
            'num_vals': torch.tensor(num_vals, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    # 必ず呼び出される関数
    def setup(self, stage):
        pass

    # Trainer.fit() 時に呼び出される
    def train_dataloader(self):
        if self.cfg.train_by_all_data:
            tr = self.cfg.train_df
        else:
            tr = self.cfg.train_df[self.cfg.train_df.fold != self.cfg.fold]

        train_ds = MlpWithNlpDataset(
            df=tr,
            cfg=self.cfg,
            phase='train'
        )
        return DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True,
                          num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn)

    # Trainer.fit() 時に呼び出される
    def val_dataloader(self):
        val = get_val(self.cfg)

        valid_ds = MlpWithNlpDataset(
            df=val,
            cfg=self.cfg,
            phase='valid'
        )
        return DataLoader(valid_ds, batch_size=self.cfg.batch_size, shuffle=False, drop_last=False,
                          num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn)
