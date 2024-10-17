import numpy as np
import pandas as pd
import json
from pathlib import Path
import pickle
from glob import glob
import cv2
from PIL import Image
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pdb import set_trace as st

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pytorch_lightning as pl
from .util import *
import os

from multiprocessing import Pool, cpu_count
import sys
import pdb

def sigmoid(x):
    return 1/(1 + np.exp(-x))

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

class MLPDataset(Dataset):
    def __init__(self, df, cfg, phase):
        self.cfg = cfg
        self.phase = phase
        if phase != 'test':
            self.labels = df[cfg.label_features].values
        self.metas = df[cfg.meta_cols].values

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = torch.FloatTensor(self.metas[idx])
        if self.phase == 'test':
            return meta

        label = self.labels[idx]
        if type(self.cfg.label_features) == list:
            return meta, torch.FloatTensor(label) # multi class
        else:
            if str(self.cfg.criterion) == 'CrossEntropyLoss()':
                return meta, label
            # return meta, torch.FloatTensor(label) # multi class
            return meta, label.astype(np.float32)

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
        # if self.cfg.upsample:
        #     dfs = []
        #     dfs.append(tr[tr['Typical Appearance']==1])
        #     for _ in range(2):
        #         dfs.append(tr[tr['Negative for Pneumonia']==1])
        #     for _ in range(3):
        #         dfs.append(tr[tr['Indeterminate Appearance']==1])
        #     for _ in range(7):
        #         dfs.append(tr[tr['Atypical Appearance']==1])
        #     tr = pd.concat(dfs)
        print('len(train):', len(tr))
        train_ds = MLPDataset(
            df=tr,
            cfg=self.cfg,
            phase='train'
        )
        return DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True,
                          num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn)

    # Trainer.fit() 時に呼び出される
    def val_dataloader(self):
        val = get_val(self.cfg)

        print('len(valid):', len(val))
        valid_ds = MLPDataset(
            df=val,
            cfg=self.cfg,
            phase='valid'
        )
        return DataLoader(valid_ds, batch_size=self.cfg.batch_size, shuffle=False, drop_last=False,
                          num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn)
