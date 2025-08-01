# src/utils/dataloader_factory.py
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.utils.data import DataLoader
import cv2
import numpy as np
from transformers import RobertaConfig, RobertaModel, XLNetTokenizer, AutoTokenizer
import torch
import sys
import os
sys.path.append(os.path.abspath(".."))
from src.lightning.data_modules.util import get_val
from src.lightning.data_modules.classification import get_dataset_class
from src.lightning.data_modules.segmentation import SegmentationDataset, SegmentationDataset3D
from src.lightning.data_modules.mlp_with_nlp import MlpWithNlpDataset
from src.lightning.data_modules.mlp import MLPDataset
from src.lightning.data_modules.nlp import NLPDataset
from src.lightning.data_modules.mil import MilDataset
from src.lightning.data_modules.gnn import Atma16Dataset

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

'''
# cfg.valid_df = None
def get_val(cfg):
    if cfg.valid_df is None: 
        val = cfg.train_df[cfg.train_df.fold == cfg.fold] # 假設現在 fold=0 -> fold0 就是驗證資料
    else:
        val = cfg.valid_df[cfg.valid_df.fold == cfg.fold]
    return val
'''
# vaild test 時使用
def prepare_classification_loader(cfg, split='val'):
    if split == 'train':
        # df = cfg.train_df[cfg.train_df.fold != cfg.fold]
        df = cfg.train_2nd_df
    elif split == 'val':  # here
        df = get_val(cfg)  # return Dataframe (above)
    elif split == 'test':
        df = cfg.test_df
    else:
        raise
    
    claz = get_dataset_class(cfg)  # 其中 claz 會指向 ClassificationDataset(data_module/classification.py) 這個類別
    ds = claz(
        df=df,  
        transforms=cfg.transform['val'],
        cfg=cfg,
        phase='test'  # only return image(no label)
    )
    return df, DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                      num_workers=cfg.n_cpu, worker_init_fn=worker_init_fn)  # 如果是 train:(image, label)

def prepare_mlp_loader(cfg, split='val'):
    if split == 'train':
        df = cfg.train_df[cfg.train_df.fold != cfg.fold]
    elif split == 'val':
        df = get_val(cfg)
    elif split == 'test':
        df = cfg.test_df
    else:
        raise

    ds = MLPDataset(
        df=df,
        cfg=cfg,
        phase='test'
    )
    return df, DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                      num_workers=cfg.n_cpu, worker_init_fn=worker_init_fn)

def prepare_gnn_loader(cfg, split='val'):
    if split == 'train':
        df = cfg.train_df[cfg.train_df.fold != cfg.fold]
    elif split == 'val':
        df = get_val(cfg)
    elif split == 'test':
        df = cfg.test_df
    else:
        raise
    if cfg.compe == 'atma16':
        ds = Atma16Dataset(
            df=df,
            cfg=cfg,
            phase='test'
        )
    return df, GeometricDataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                      num_workers=cfg.n_cpu, worker_init_fn=worker_init_fn)

def prepare_mlp_with_nlp_loader(cfg, split='val'):
    if split == 'train':
        df = cfg.train_df[cfg.train_df.fold != cfg.fold]
    elif split == 'val':
        df = get_val(cfg)
    elif split == 'test':
        df = cfg.test_df
    else:
        raise

    ds = MlpWithNlpDataset(
        df=df,
        cfg=cfg,
        phase='test'
    )
    return df, DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                      num_workers=cfg.n_cpu, worker_init_fn=worker_init_fn)

def prepare_seg_loader(cfg, split='val'):
    if split == 'train':
        df = cfg.train_df[cfg.train_df.fold != cfg.fold]
    elif split == 'val':
        df = get_val(cfg)
    elif split == 'test':
        df = cfg.test_df
    else:
        raise

    claz = SegmentationDataset3D if cfg.seg_3d else SegmentationDataset
    ds = claz(
        df=df,
        transforms=cfg.transform['val'],
        cfg=cfg,
        phase='test'
    )
    return df, DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                      num_workers=cfg.n_cpu, worker_init_fn=worker_init_fn)

def prepare_nlp_loader(cfg, split='test'):
    if split == 'train':
        df = cfg.train_df[cfg.train_df.fold != cfg.fold]
    elif split == 'val':
        df = get_val(cfg)
    elif split == 'test':
        df = cfg.test_df
    else:
        raise
    ds = NLPDataset(
        df=df,
        cfg=cfg,
        phase='test',
    )
    return df, DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                      num_workers=cfg.n_cpu, worker_init_fn=worker_init_fn)

def prepare_effdet_loader(cfg, predict_valid=False):
    if predict_valid:
        df = cfg.train_df[cfg.train_df.fold == cfg.fold]
        image = cv2.imread(path)[:,:,::-1]

    ds = EffdetDatasetTest(
        df=df,
        transforms=cfg.transform['test'],
        cfg=cfg
    )
    return df, DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                      num_workers=cfg.n_cpu, worker_init_fn=worker_init_fn)
