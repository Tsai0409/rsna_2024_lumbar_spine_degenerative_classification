import numpy as np
import os
from pathlib import Path
from typing import Optional
from multiprocessing import Pool, cpu_count

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
from torch.utils.data import Dataset#, DataLoader

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

def unique_last(seq: list[int]) -> tuple[list[int], list[int]]:
    """複数回出現する要素は最後のみ残す

    Args:
        seq (list[int]): 入力シーケンス

    Returns:
        tuple[int, int]: 重複を除いたシーケンス, 重複を除いたシーケンスの出現回数
    """
    out = []
    out_cnt = []
    seen = set()
    for i in seq[::-1]:
        if i not in seen:
            out.append(i)
            seen.add(i)
            out_cnt.append(1)
        else:
            index = out.index(i)
            out_cnt[index] += 1

    return out[::-1], out_cnt[::-1]

# class Atma16Dataset(InMemoryDataset):
#     """Atma16Dataset
#     InMemoryDatasetを継承して、データセットを作成
#     一回作成すると、graph_cache_pathにキャッシュされる
#     再度作成する場合は、graph_cache_pathを削除する
#     参考: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html#creating-in-memory-datasets
#     """

#     def __init__(
#         self,
#         df,
#         cfg,
#         phase='train',
#     ):
#         self.cfg = cfg
#         self.df = df
#         self.phase = phase
#         self.G = cfg.G
#         self.k = cfg.k
#         if cfg.clear_cache:
#             if os.path.exists(cfg.graph_cache_path):
#                 print('\n\ncache clear.\n')
#                 os.system(f'rm -r {cfg.graph_cache_path}')
#         # print(self.processed_paths[0])
#         super(Atma16Dataset, self).__init__(cfg.graph_cache_path, transform=None, pre_transform=None)
#         self.load(self.processed_paths[0])

#     @property
#     def processed_file_names(self):
#         # このメソッドは、処理済みデータファイルの名前のリストを返します
#         return [f'{self.phase}_{p}' for p in self.cfg.processed_file_paths]


#     def exec(self, arg):
#         session_id, idf = arg
#         seq = idf['yad_no'].values.tolist()
#         data = self.create_subgraph_data(session_id, seq)
#         if self.phase != 'test':
#             label = idf['target'].values[0]
#             data.y = (data.subset_node_idx == label).float()
#             data.label = label
#         return data



#     def process(self):
#         self.df = self.df.sort_values('seq_no')

#         # p = Pool(processes=cpu_count())
#         # data_list = []
#         # df_list = list(self.df.groupby('session_id'))
#         # with tqdm(total=len(df_list)) as pbar:
#         #     for res in p.imap(self.exec, df_list):
#         #         data_list.append(res)
#         #         pbar.update(1)
#         # p.close()


#         data_list = []
#         for session_id, idf in tqdm(self.df.groupby('session_id')):
#             seq = idf['yad_no'].values.tolist()
#             data = self.create_subgraph_data(session_id, seq)
#             if self.phase != 'test':
#                 label = idf['target'].values[0]
#                 data.y = (data.subset_node_idx == label).float()
#                 data.label = label
#             data_list.append(data)

#         self.save(data_list, self.processed_paths[0])

#     def create_subgraph_data(self, session_id: str, seq: list[int]) -> Data:
#         # サブグラフデータの作成
#         # seq: 訪問済みノード
#         seq, seq_cnt = unique_last(seq)  # 複数回訪問したノードは最後のみ残す
#         node_idx = torch.tensor(seq, dtype=torch.long)
#         subset, subset_edge_index, mapping, edge_mask = k_hop_subgraph(
#             node_idx=node_idx,
#             num_hops=self.k,
#             edge_index=self.G.edge_index,
#             relabel_nodes=True,
#             flow="target_to_source",
#         )

#         # edge特徴量
#         edge_attr = self.G.edge_attr[edge_mask]  # (E, 1)
#         # 訪問済みノードに対応するエッジのみ1
#         connected = torch.isin(subset_edge_index, mapping).float().T  # (E, 2)
#         # 訪問済みノードから訪問済みノードへのエッジのみ1
#         seq_edge = connected.prod(dim=1).view(-1, 1)  # (E, 1)
#         edge_attr = torch.cat([edge_attr, connected, seq_edge], dim=1)  # (E, 4)

#         # node特徴量
#         x: torch.Tensor = self.G.x[subset]
#         num_node: int = x.shape[0]
#         # 最後のノードは1, それ以外は0
#         is_last = torch.zeros(num_node).float()
#         is_last[mapping[-1]] = 1.0
#         # 訪問済みノードは1, 未訪問ノードは0
#         is_visited = torch.zeros(num_node).float()
#         is_visited[mapping] = 1.0
#         # 訪問順
#         order_of_visit = torch.zeros((num_node)).float()
#         order_of_visit[mapping] = torch.arange(len(mapping)).float() + 1.0
#         # 奇数番目のノードは1, 偶数番目のノードは0
#         is_odd = torch.zeros((x.shape[0])).long()
#         is_odd[mapping] = torch.arange(1, len(mapping) + 1).long()
#         is_odd = (is_odd % 2).float()
#         # 訪問回数
#         visit_cnt = torch.zeros(num_node).float()
#         visit_cnt[mapping] = torch.tensor(seq_cnt).float()

#         x = torch.cat(
#             [
#                 x,
#                 is_last.view(-1, 1),
#                 is_visited.view(-1, 1),
#                 torch.log1p(order_of_visit.view(-1, 1)),
#                 is_odd.view(-1, 1),
#                 torch.log1p(visit_cnt.view(-1, 1)),
#             ],
#             dim=1,
#         )

#         return Data(
#             x=x.float(),
#             edge_index=subset_edge_index,
#             edge_attr=edge_attr,
#             subset_node_idx=subset,
#             label=None,
#             session_id=session_id,
#         )

class Atma16Dataset(Dataset):
    def __init__(self, df, cfg, phase='train'):
        self.cfg = cfg
        self.df = df.sort_values('seq_no')
        self.phase = phase
        self.G = cfg.G
        self.k = cfg.k
        self.grouped = self.df.groupby('session_id')
        self.session_ids = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.session_ids)

    def __getitem__(self, idx):
        session_id = self.session_ids[idx]
        idf = self.grouped.get_group(session_id)
        seq = idf['yad_no'].values.tolist()
        data = self.create_subgraph_data(session_id, seq)
        if self.phase != 'test':
            label = idf['target'].values[0]
            data.y = (data.subset_node_idx == label).float()
            data.label = label
        return data

    def create_subgraph_data(self, session_id: str, seq: list[int]) -> Data:
        # サブグラフデータの作成
        # seq: 訪問済みノード
        seq, seq_cnt = unique_last(seq)  # 複数回訪問したノードは最後のみ残す
        node_idx = torch.tensor(seq, dtype=torch.long)
        subset, subset_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=self.k,
            edge_index=self.G.edge_index,
            relabel_nodes=True,
            flow="target_to_source",
        )

        # edge特徴量
        edge_attr = self.G.edge_attr[edge_mask]  # (E, 1)
        # 訪問済みノードに対応するエッジのみ1
        connected = torch.isin(subset_edge_index, mapping).float().T  # (E, 2)
        # 訪問済みノードから訪問済みノードへのエッジのみ1
        seq_edge = connected.prod(dim=1).view(-1, 1)  # (E, 1)
        edge_attr = torch.cat([edge_attr, connected, seq_edge], dim=1)  # (E, 4)

        # node特徴量
        x: torch.Tensor = self.G.x[subset]
        num_node: int = x.shape[0]
        # 最後のノードは1, それ以外は0
        is_last = torch.zeros(num_node).float()
        is_last[mapping[-1]] = 1.0
        # 訪問済みノードは1, 未訪問ノードは0
        is_visited = torch.zeros(num_node).float()
        is_visited[mapping] = 1.0
        # 訪問順
        order_of_visit = torch.zeros((num_node)).float()
        order_of_visit[mapping] = torch.arange(len(mapping)).float() + 1.0
        # 奇数番目のノードは1, 偶数番目のノードは0
        is_odd = torch.zeros((x.shape[0])).long()
        is_odd[mapping] = torch.arange(1, len(mapping) + 1).long()
        is_odd = (is_odd % 2).float()
        # 訪問回数
        visit_cnt = torch.zeros(num_node).float()
        visit_cnt[mapping] = torch.tensor(seq_cnt).float()

        x = torch.cat(
            [
                x,
                is_last.view(-1, 1),
                is_visited.view(-1, 1),
                torch.log1p(order_of_visit.view(-1, 1)),
                is_odd.view(-1, 1),
                torch.log1p(visit_cnt.view(-1, 1)),
            ],
            dim=1,
        )

        return Data(
            x=x.float(),
            edge_index=subset_edge_index,
            edge_attr=edge_attr,
            subset_node_idx=subset,
            label=None,
            session_id=session_id,
        )

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_dataset_class(cfg):
    if cfg.compe == 'atma16':
        claz = Atma16Dataset
    else:
        raise
    return claz

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
        claz = get_dataset_class(self.cfg)
        train_ds = claz(
            cfg=self.cfg,
            df=tr,
            phase='train',
        )
        return DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.n_cpu,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self):
        claz = get_dataset_class(self.cfg)
        val = get_val(self.cfg)
        valid_ds = claz(
            cfg=self.cfg,
            df=val,
            phase='valid',
        )
        return DataLoader(
            valid_ds,
            batch_size=self.cfg.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.n_cpu,
            worker_init_fn=worker_init_fn
        )
