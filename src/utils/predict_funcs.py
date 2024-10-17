from scipy.special import softmax
import torch
from tqdm import tqdm
import numpy as np
from pdb import set_trace as st
import torchvision.transforms.functional as F
from torch_geometric.utils import degree
import os
import joblib
import pickle
def pickle_dump(data, path):
    with open(path, mode='wb') as f:
        pickle.dump(data, f)

# with torch.no_grad():
#     with torch.cuda.amp.autocast(enabled=use_amp):

def classification_predict(cfg, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.model.eval()
    tta_predictions = [[] for _ in range(len(cfg.add_imsizes_when_inference)*cfg.tta)]
    assert cfg.tta <= 8
    for images_n, images in enumerate(tqdm(loader)):
        with torch.no_grad():
            tta_n = 0
            if getattr(cfg, 'multi_image_4classes', False):
                images = [i.to(device) for i in images]
            elif len(cfg.meta_cols) != 0:
                images, meta = images
                images, meta = images.to(device), meta.to(device)
            else:
                try:
                    images = images.to(device)
                except:
                    images = [i.to(device) for i in images]

            for add_imsize in cfg.add_imsizes_when_inference:
                predictions = []
                features_list = []
                if (add_imsize[0] != 0) | (add_imsize[1] != 0):
                    images = F.resize(img=images, size=(cfg.image_size[0]+add_imsize[0], cfg.image_size[1]+add_imsize[1]))
                if images_n == 0:
                    try:
                        print('images.size():', images.size())
                    except:
                        pass
                for flip_tta_n in range(cfg.tta):
                    if flip_tta_n % 2 == 1:
                        images = torch.flip(images, (3,))
                    if flip_tta_n in [2, 6]:
                        images = torch.flip(images, (2,))
                    if flip_tta_n == 4:
                        images = torch.transpose(images, 2,3)

                    input_ = images if len(cfg.meta_cols) == 0 else (images, meta)
                    with torch.cuda.amp.autocast(enabled=cfg.inf_fp16):
                        if cfg.output_features:
                            pred, features = cfg.model.extract_with_features(input_)
                        else:
                            pred = cfg.model(input_)
                            if isinstance(pred, tuple):
                                pred = pred[0]
                        if cfg.output_features:
                            features_list.append(features.detach())
                    tta_predictions[tta_n] += pred.detach().cpu().numpy().tolist()
                    # tta_predictions[tta_n].append(torch.cat(predictions))
                    tta_n += 1
    if cfg.output_features:
        return tta_predictions, torch.cat(features_list).cpu().numpy()
    else:
        return tta_predictions

def mlp_with_nlp_predict(cfg, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.model.eval()
    with torch.no_grad():
        tta_predictions = []
        for tta_n in range(cfg.tta):
            predictions = []
            for images in tqdm(loader):
                images = images.to(device)
                if tta_n % 2 == 1:
                    images = torch.flip(images, (3,))
                if tta_n % 4 >= 2:
                    images = torch.flip(images, (2,))
                if tta_n % 8 >= 4:
                    images = torch.transpose(images, 2,3)
                pred = cfg.model(images)
                # features_list.append(features.detach())
                predictions.append(pred.detach())
            tta_predictions.append(torch.cat(predictions).cpu().numpy())
    return tta_predictions

def mlp_predict(cfg, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.model.eval()
    with torch.no_grad():
        tta_predictions = []
        for tta_n in range(cfg.tta):
            predictions = []
            for metas in tqdm(loader):
                metas = metas.to(device)
                pred = cfg.model(metas)
                predictions.append(pred.detach())
            tta_predictions.append(torch.cat(predictions).cpu().numpy())
    return tta_predictions

# def decode(pred: torch.Tensor, batch: Batch) -> torch.Tensor:
#     sizes = degree(batch.batch, dtype=torch.long).tolist()
#     logit_list = pred.split(sizes)
#     subset_node_idx_list = batch.subset_node_idx.split(sizes)
#     preds = []
#     for y_pred, subset_node_idx in zip(logit_list, subset_node_idx_list):
#         prob = y_pred
#         # 確率が大きい上位10個を取得
#         arg_idx = torch.argsort(prob, descending=True)
#         arg_topk = arg_idx[:10]
#         topk_node_idx = subset_node_idx[arg_topk]
#         # 10個未満の場合は0を追加
#         if len(topk_node_idx) < 10:
#             topk_node_idx = torch.cat(
#                 [
#                     topk_node_idx,
#                     torch.zeros(
#                         10 - len(topk_node_idx), dtype=torch.long, device=topk_node_idx.device
#                     ),
#                 ]
#             )
#         preds.append(topk_node_idx)
#     return torch.cat(preds)
def gnn_predict(cfg, loader, output_path, split):
    os.system(f'mkdir -p {output_path}/{split}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.model.eval()
    id_pred_item_map = {}
    id_pred_logit_map = {}
    for batch in tqdm(loader):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg.inf_fp16):
                batch = batch.to(device)
                pred = cfg.model(batch)

                sizes = degree(batch.batch, dtype=torch.long).tolist()
                logit_list = pred.split(sizes)
                subset_node_idx_list = batch.subset_node_idx.split(sizes)

                assert len(logit_list) == len(batch.session_id) == len(subset_node_idx_list)
                for logits, items, session_id in zip(logit_list, subset_node_idx_list, batch.session_id):
                    np.save(f'{output_path}/{split}/{session_id}___items.npy', items.detach().cpu().numpy().tolist())
                    np.save(f'{output_path}/{split}/{session_id}___logits.npy', logits.detach().cpu().numpy().tolist())

    return id_pred_item_map, id_pred_logit_map

def seg_predict(cfg, loader, save_dir=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.model.eval()
    with torch.no_grad():
        tta_predictions = []
        for tta_n in range(cfg.tta):
            predictions = []
            for images, ids in tqdm(loader):
                images = images.to(device)
                if tta_n % 2 == 1:
                    images = torch.flip(images, (3,))
                if tta_n % 4 >= 2:
                    images = torch.flip(images, (2,))
                if tta_n % 8 >= 4:
                    images = torch.transpose(images, 2,3)
                pred = cfg.model(images)
                if tta_n % 2 == 1:
                    pred = torch.flip(pred, (3,))
                if tta_n % 4 >= 2:
                    pred = torch.flip(pred, (2,))
                if tta_n % 8 >= 4:
                    pred = torch.transpose(pred, 2,3)
                # features_list.append(features.detach())
                # predictions.append(pred.detach().cpu())
                if cfg.save_preds:
                    for pr, id in zip(pred.detach().cpu().numpy(), ids):
                        np.save(f'{save_dir}/{id}.npy', pr)
                # if cfg.save_targets:
                #     for im, id in zip(masks.detach().cpu().numpy(), ids):
                #         np.save(f'{save_dir.replace('preds', 'targets')}/{id}.npy', im)

            # tta_predictions.append(torch.cat(predictions).numpy())
    # return np.mean(tta_predictions, axis=0)

def seg_predict_calc_metric(cfg, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.model.eval()
    with torch.no_grad():
        tta_predictions = []
        targets = []
        for tta_n in range(cfg.tta):
            predictions = []
            for images, masks in tqdm(loader):
                images = images.to(device)
                if tta_n % 2 == 1:
                    images = torch.flip(images, (3,))
                if tta_n % 4 >= 2:
                    images = torch.flip(images, (2,))
                if tta_n % 8 >= 4:
                    images = torch.transpose(images, 2,3)
                pred = cfg.model(images)
                if tta_n % 2 == 1:
                    pred = torch.flip(pred, (3,))
                if tta_n % 4 >= 2:
                    pred = torch.flip(pred, (2,))
                if tta_n % 8 >= 4:
                    pred = torch.transpose(pred, 2,3)
                # features_list.append(features.detach())
                predictions.append(pred.detach().cpu())
                if tta_n == 0:
                    targets.append(masks)
            tta_predictions.append(torch.cat(predictions))
    targets = torch.cat(targets)
    preds = torch.mean(tta_predictions, axis=0)
    # score = cfg.metric(targets, preds)
    # print('score:', score)

    return preds.numpy(), targets.numpy()

def metric_learning_mlp_with_nlp_predict(cfg, loader, fliplr=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.model.eval()
    with torch.no_grad():
        features = []
        for input in tqdm(loader):
            ids = input['ids'].to(device, dtype=torch.long)
            mask = input['mask'].to(device, dtype=torch.long)
            num_vals = input['num_vals'].to(device, dtype=torch.float)
            feature = cfg.model.extract(ids, mask, num_vals)
            features.append(feature.detach())
    return torch.cat(features).cpu().numpy()

def nlp_predict(cfg, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.model.eval()
    with torch.no_grad():
        predictions = []
        features_list = []
        for ids, masks, token_type_ids in tqdm(loader):
            ids, masks, token_type_ids = ids.to(device), masks.to(device), token_type_ids.to(device)
            if cfg.output_features:
                pred, features = cfg.model.forward_with_features(ids, masks, token_type_ids)
            else:
                pred = cfg.model(ids, masks, token_type_ids)
            predictions.append(pred.detach())
            if cfg.output_features:
                features_list.append(features.detach())
    if cfg.output_features:
        return torch.cat(predictions).cpu().numpy(), torch.cat(features_list).cpu().numpy()
    return torch.cat(predictions).cpu().numpy()

def effdet_predict(cfg, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.model.eval()
    with torch.no_grad():
        tta_predictions = []
        for tta_n in range(cfg.tta):
            predictions = []
            for images in tqdm(loader):
                images = images.to(device)
                # if tta_n % 2 == 1:
                #     images = torch.flip(images, (3,))
                # if tta_n % 4 >= 2:
                #     images = torch.flip(images, (2,))
                # if tta_n % 8 >= 4:
                #     images = torch.transpose(images, 2,3)
                pred = cfg.model(images)
                # features_list.append(features.detach())
                predictions.append(pred.detach())
            return torch.cat(predictions).cpu().numpy()
            tta_predictions.append(torch.cat(predictions).cpu().numpy())
    return tta_predictions
