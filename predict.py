# predict.py
import torch
import random
import os
import numpy as np
import pandas as pd
import gc
import argparse
import warnings
warnings.simplefilter('ignore')
from multiprocessing import cpu_count
from pdb import set_trace as st
from src.utils.google_spread_sheet_editor import GoogleSpreadSheetEditor

def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, default='Test', help="config name in configs.py")
    parser.add_argument("--type", '-t', type=str, default='classification', help="type")
    parser.add_argument("--debug", action='store_true', help="debug")
    parser.add_argument("--fold", '-f', type=int, default=0, help="fold num")
    return parser.parse_args()

def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_args()

    if args.type == 'classification':  # default='classification'
        from src.configs import *
    elif args.type == 'effdet':
        from src.effdet_configs import *
    elif args.type == 'seg':
        from src.seg_configs import *
    elif args.type == 'nlp':
        from src.nlp_configs import *
    elif args.type == 'mlp_with_nlp':
        from src.mlp_with_nlp_configs import *
    elif args.type == 'mlp':
        from src.mlp_configs import *
    elif args.type == 'gnn':
        from src.gnn_configs import *

    try:
        cfg = eval(args.config)(args.fold)
    except:
        cfg = eval(args.config)()

    cfg.fold = args.fold
    if cfg.batch_size != 1:  # self.batch_size = 16
        cfg.batch_size *= 2   # cfg.batch_size = 32

    # self.predict_valid = True
    if (not cfg.predict_valid) & (not cfg.predict_test):
        print('(not cfg.predict_valid) & (not cfg.predict_test)!')
        exit()

    # RESULTS_PATH_BASE = f'results'
    RESULTS_PATH_BASE = '/kaggle/working/duplicate/ckpted'  # (ckpted 放在 kaggle 上的權重檔)

    from src.utils.predict_funcs import classification_predict as predict
    from src.utils.dataloader_factory import prepare_classification_loader as prepare_loader
    
    if args.debug:
        cfg.n_cpu = 1
        n_gpu = 1
    else:
        n_gpu = torch.cuda.device_count()
        cfg.n_cpu = np.min([cpu_count(), cfg.batch_size])

    if args.type not in ['nlp', 'mlp_with_nlp', 'mlp', 'gnn']:
        if type(cfg.image_size) == int:
            cfg.image_size = (cfg.image_size, cfg.image_size)
        cfg.transform = cfg.transform(cfg.image_size)

    seed_everything()

    # self.force_use_model_path_config_when_inf = None
    if getattr(cfg, 'force_use_model_path_config_when_inf', False):  # get attribute 看看是否有這個屬性；預設為False
        load_model_config_dir = f'{RESULTS_PATH_BASE}/{cfg.force_use_model_path_config_when_inf}'
        # cfg.force_use_model_path_config_when_inf = None
    else:  # here
        load_model_config_dir = f'{RESULTS_PATH_BASE}/{args.config}'  # load_model_config_dir = /kaggle/working/duplicate/ckpted/rsna_sagittal_level_cl_spinal_v1

#    OUTPUT_PATH = f'{RESULTS_PATH_BASE}/{args.config}'
    OUTPUT_PATH = f'/kaggle/working/ckpt/{args.config}'  # OUTPUT_PATH = /kaggle/working/ckpt/rsna_sagittal_level_cl_spinal_v1 (ckpt 執行完產生出來的)
    os.system(f'mkdir -p {OUTPUT_PATH}')  # make directory 建立目錄

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # self.use_last_ckpt_when_inference = True
    if getattr(cfg, 'use_last_ckpt_when_inference', False):  
        file_name_base = 'last_fold'
    else:
        file_name_base = 'fold_'
    
    state_dict_path = f'{load_model_config_dir}/{file_name_base}{args.fold}.ckpt'  
    # state_dict_path = /kaggle/working/duplicate/ckpted/rsna_sagittal_level_cl_spinal_v1/last_fold0.ckpt

    # self.no_trained_model_when_inf = False
    if not getattr(cfg, 'no_trained_model_when_inf', False):  # cfg 中的 'no_trained_model_when_inf' 這個屬性為 False，if not 所以 False 成立，進入 if 迴圈
        try:
            state_dict = torch.load(state_dict_path)['state_dict']  # 如果 checkpoint 檔案儲存的是一個字典，裡面有一個 col 叫做 'state_dict'
        except:  # here
            state_dict = torch.load(state_dict_path)
            # state_dict 是一個 dictionary，其結構通常如下：
            # key：每個鍵都是字串，對應模型中某個參數或緩衝區的名稱（例如 "layer1.weight"、"layer1.bias" 等）
            # value：每個值是 torch.Tensor，存儲了該參數或緩衝區的數值

        torch_state_dict = {}
        delete_model_model = True
        delete_model = True

        for k, v in state_dict.items():  # (key, value)
            if not k.startswith('model.model.'):
                delete_model_model = False
            if not k.startswith('model.'):
                delete_model = False

        for k, v in state_dict.items():
            if delete_model_model:
                torch_state_dict[k[12:]] = v  # 如果 key 是以 model.model. 開頭的樣式來存放，將 k[0:11] 的部分刪除；並以 (key, value) 存放在 torch_state_dict 字典中
            elif delete_model:
                torch_state_dict[k[6:]] = v
            else:
                torch_state_dict[k] = v

        print(f'load model weight from checkpoint: {state_dict_path}')

    # self.no_trained_model_when_inf = False
    if not getattr(cfg, 'no_trained_model_when_inf', False):  # 進入 if 迴圈；在推論時不使用訓練好的模型(在推論時不會進來)
        cfg.model.load_state_dict(torch_state_dict)  # 載入模型權重
    cfg.model.to(device)

    # self.predict_valid = True
    if cfg.predict_valid:  # 驗證集的預測
        val, val_loader = prepare_loader(cfg, split='val')  # val：通常是一個 DataFrame，包含驗證集的原始資料（例如 ID、標籤等）；val_loader：是一個資料載入器（DataLoader），可以依批次讀取驗證集的資料，方便後續推論使用。
        preds = predict(cfg, val_loader)
        pred_cols = [f'pred_{c}' for c in cfg.label_features]  # self.label_features = ['l1_spinal', 'l2_spinal', 'l3_spinal', 'l4_spinal', 'l5_spinal']
        val[pred_cols] = preds[0]
        val.to_csv(f'{OUTPUT_PATH}/oof_fold{args.fold}.csv', index=False)  # /kaggle/working/ckpt/rsna_sagittal_level_cl_spinal_v1/oof_fold0.csv
        print(f'val save to {OUTPUT_PATH}/oof_fold{args.fold}.csv')

    # class rsna_sagittal_level_cl_spinal_v1、class rsna_sagittal_level_cl_nfn_v1 -> self.predict_test = True
    # class rsna_sagittal_cl -> self.predict_test = False
    if cfg.predict_test:
        test, test_loader = prepare_loader(cfg, split='test')
        preds = predict(cfg, test_loader)
        preds_n = 0
        for add_imsizes_n, add_imsizes in enumerate(cfg.add_imsizes_when_inference):
            for tta_n in range(cfg.tta):
                if add_imsizes_n == 0:
                    suffix = ''
                else:
                    suffix = f'multi_scale_{add_imsizes_n}_'
                if tta_n != 0:
                    suffix += f'flip_{tta_n}'
                if suffix == '':
                    pred_cols = [f'pred_{c}' for c in cfg.label_features]
                else:
                    pred_cols = [f'pred_{c}_{suffix}' for c in cfg.label_features]
                test[pred_cols] = preds[preds_n]
                preds_n += 1

        test.to_csv(f'{OUTPUT_PATH}/test_fold{args.fold}.csv', index=False)  # /kaggle/working/ckpt/rsna_sagittal_level_cl_spinal_v1/test_fold0.csv
        print(f'test save to {OUTPUT_PATH}/test_fold{args.fold}.csv')

print('predict.py finish')