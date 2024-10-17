import numpy as np
import pandas as pd
import json
from pathlib import Path
import pickle
from glob import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from pdb import set_trace as st

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .util import *
from transformers import RobertaConfig, RobertaModel, XLNetTokenizer, AutoTokenizer, GPT2Tokenizer

tokenizer_path_base = '/groups/gca50041/ariyasu/nlp_models/'
model_tokenizer_map = {
    'distilroberta/base': 'distilroberta/base',
    'bart/bart-base': 'roberta/roberta-base',
    'bart/bart-large': 'roberta/roberta-large',
    'roberta/roberta-base': 'roberta/roberta-base',
    'roberta/roberta-large': 'roberta/roberta-large',
    'electra/base-discriminator': 'electra/base-discriminator',
    'electra/large-discriminator': 'electra/large-discriminator',
    'deberta/deberta-v3-small': '/deberta/deberta-v3-small',
    'deberta/deberta-v3-xsmall': '/deberta/deberta-v3-xsmall',
    'deberta/base': '/deberta/base',
    'deberta/large': '/deberta/large',
    'deberta/v2-xlarge': '/deberta/v2-xlarge',
    'deberta/v2-xxlarge': '/deberta/v2-xxlarge',
    'albert_v2/xxlarge': '/albert_v2/xxlarge',
    'xlnet/xlnet-base-cased-pytorch_model.bin': 'xlnet/xlnet-base-cased-spiece.model',
    'xlnet/xlnet-large-cased-pytorch_model.bin': 'xlnet/xlnet-large-cased-spiece.model',
}

class NLPDatasetWithMeta:
    def __init__(self, df, cfg):
        tokenizer_path = model_tokenizer_map[cfg.model_path] if cfg.model_path in model_tokenizer_map else cfg.model_path
        if 'xlnet' in cfg.model_path:
            self.tokenizer = XLNetTokenizer.from_pretrained(tokenizer_path_base + tokenizer_path)
        elif 'gpt2' in cfg.model_path:
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                tokenizer_path_base + cfg.model_path,
                bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_base + tokenizer_path)
        self.df = df
        self.cfg = cfg
        self.meta_cols = df[cfg.meta_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df[self.cfg.text_feature].values[idx])
        inputs = self.tokenizer(
            text,
            max_length=self.cfg.max_len,
            padding="max_length",
            truncation=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        targets = self.df[self.cfg.label_features].values[idx]
        # aux = self.df["aux_target"].values[idx] + 4

        # aux_targets = np.zeros(7, dtype=float)
        # aux_targets[aux] = 1.0

        numerical_features = self.meta_cols[idx]

        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(numerical_features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

class NLPDataset:
    def __init__(self, df, cfg, phase='train'):
        tokenizer_path = model_tokenizer_map[cfg.model_path] if cfg.model_path in model_tokenizer_map else cfg.model_path
        if 'xlnet' in cfg.model_path:
            self.tokenizer = XLNetTokenizer.from_pretrained(tokenizer_path_base + tokenizer_path)
        elif 'gpt2' in cfg.model_path:
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                tokenizer_path_base + cfg.model_path,
                bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_base + tokenizer_path)
        self.df = df
        self.cfg = cfg
        self.phase = phase
        # self.meta_cols = df[cfg.meta_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df[self.cfg.text_feature].values[idx])
        inputs = self.tokenizer(
            text,
            max_length=self.cfg.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        if self.phase == 'test':
            return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(token_type_ids, dtype=torch.long)
        targets = self.df[self.cfg.label_features].values[idx]
        # aux = self.df["aux_target"].values[idx] + 4

        # aux_targets = np.zeros(7, dtype=float)
        # aux_targets[aux] = 1.0

        # numerical_features = self.meta_cols[idx]
        if type(self.cfg.label_features) == list:
            return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(token_type_ids, dtype=torch.long), torch.tensor(targets, dtype=torch.float32)
        else:
            return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(token_type_ids, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

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

        train_ds = NLPDataset(
            df=tr,
            cfg=self.cfg,
        )
        return DataLoader(train_ds, batch_size=self.cfg.batch_size, pin_memory=True, shuffle=True, drop_last=True,
                          num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn)

    # Trainer.fit() 時に呼び出される
    def val_dataloader(self):
        val = get_val(self.cfg)

        valid_ds = NLPDataset(
            df=val,
            cfg=self.cfg,
        )
        return DataLoader(valid_ds, batch_size=self.cfg.batch_size, pin_memory=True, shuffle=False, drop_last=False,
                          num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn)
