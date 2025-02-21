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

    if args.type == 'classification':
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
    if cfg.batch_size != 1:
        cfg.batch_size *= 2

    if (not cfg.predict_valid) & (not cfg.predict_test):
        print('(not cfg.predict_valid) & (not cfg.predict_test)!')
        exit()

    # RESULTS_PATH_BASE = f'results'
    RESULTS_PATH_BASE = '/kaggle/working/duplicate/ckpted'

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

    if getattr(cfg, 'force_use_model_path_config_when_inf', False):
        load_model_config_dir = f'{RESULTS_PATH_BASE}/{cfg.force_use_model_path_config_when_inf}'
    else:
        load_model_config_dir = f'{RESULTS_PATH_BASE}/{args.config}'

#    OUTPUT_PATH = f'{RESULTS_PATH_BASE}/{args.config}'
    OUTPUT_PATH = '/kaggle/working/ckpted/{args.config}'
    os.system(f'mkdir -p {OUTPUT_PATH}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if getattr(cfg, 'use_last_ckpt_when_inference', False):
        file_name_base = 'last_fold'
    else:
        file_name_base = 'fold_'
    state_dict_path = f'{load_model_config_dir}/{file_name_base}{args.fold}.ckpt'

    if not getattr(cfg, 'no_trained_model_when_inf', False):
        try:
            state_dict = torch.load(state_dict_path)['state_dict']
        except:
            state_dict = torch.load(state_dict_path)

        torch_state_dict = {}
        delete_model_model = True
        delete_model = True
        for k, v in state_dict.items():
            if not k.startswith('model.model.'):
                delete_model_model = False
            if not k.startswith('model.'):
                delete_model = False

        for k, v in state_dict.items():
            if delete_model_model:
                torch_state_dict[k[12:]] = v
            elif delete_model:
                torch_state_dict[k[6:]] = v
            else:
                torch_state_dict[k] = v

        print(f'load model weight from checkpoint: {state_dict_path}')
    if not getattr(cfg, 'no_trained_model_when_inf', False):
        cfg.model.load_state_dict(torch_state_dict)
    cfg.model.to(device)

    if cfg.predict_valid:
        val, val_loader = prepare_loader(cfg, split='val')
        preds = predict(cfg, val_loader)
        pred_cols = [f'pred_{c}' for c in cfg.label_features]
        val[pred_cols] = preds[0]
        val.to_csv(f'{OUTPUT_PATH}/oof_fold{args.fold}.csv', index=False)
        print(f'val save to {OUTPUT_PATH}/oof_fold{args.fold}.csv')

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

        test.to_csv(f'{OUTPUT_PATH}/test_fold{args.fold}.csv', index=False)
        print(f'test save to {OUTPUT_PATH}/test_fold{args.fold}.csv')
