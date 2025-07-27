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
        cfg = eval(args.config)(args.fold)  # here
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
    
    # self.use_last_ckpt_when_inference = True (all condition)
    if getattr(cfg, 'use_last_ckpt_when_inference', False):  # here
        file_name_base = 'last_fold'  # 載入 epoch 最後執行的權重檔
    else:
        file_name_base = 'fold_'
    
    state_dict_path = f'{load_model_config_dir}/{file_name_base}{args.fold}.ckpt'
    # state_dict_path = /kaggle/working/duplicate/ckpted/rsna_sagittal_level_cl_spinal_v1/last_fold0.ckpt

    # self.no_trained_model_when_inf = False (all condition)
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

        for k, v in state_dict.items():  # (key, value) (not use)
            if not k.startswith('model.model.'):
                delete_model_model = False
            if not k.startswith('model.'):
                delete_model = False

        for k, v in state_dict.items():  # (not use)
            if delete_model_model:
                torch_state_dict[k[12:]] = v  # 如果 key 是以 model.model. 開頭的樣式來存放，將 k[0:11] 的部分刪除；並以 (key, value) 存放在 torch_state_dict 字典中
            elif delete_model:
                torch_state_dict[k[6:]] = v
            else:
                torch_state_dict[k] = v

        print(f'load model weight from checkpoint: {state_dict_path}')

    # self.no_trained_model_when_inf = False (all condition)
    if not getattr(cfg, 'no_trained_model_when_inf', False):  # 進入 if 迴圈；在推論時不使用訓練好的模型(在推論時不會進來)
        cfg.model.load_state_dict(torch_state_dict)  # 載入模型權重
    cfg.model.to(device)

    # self.predict_valid = True (all condition)
    if cfg.predict_valid:  # 驗證集的預測；val_dataloader 定義在 data_module/classificaiton.py 中；val_loader 是以(image, label) 的資料型態回傳
        val, val_loader = prepare_loader(cfg, split='val')  # val: DataFrame，包含驗證集的原始資料（例如 ID、標籤等）；val_loader:DataLoader，可以依批次讀取驗證集的資料，方便後續推論使用。
        preds = predict(cfg, val_loader)
        pred_cols = [f'pred_{c}' for c in cfg.label_features]  # self.label_features = ['l1_spinal', 'l2_spinal', 'l3_spinal', 'l4_spinal', 'l5_spinal']
        val[pred_cols] = preds[0]
        val.to_csv(f'{OUTPUT_PATH}/oof_fold{args.fold}.csv', index=False)  # /kaggle/working/ckpt/rsna_sagittal_level_cl_spinal_v1/oof_fold0.csv
        print(f'val save to {OUTPUT_PATH}/oof_fold{args.fold}.csv')

    # slice estimation ->
        # class rsna_sagittal_level_cl_spinal_v1、class rsna_sagittal_level_cl_nfn_v1 -> self.predict_test = True
        # class rsna_sagittal_cl -> self.predict_test = False
    # axial, sagittal classification ->
        # self.predict_test = False
    if cfg.predict_test:
        test, test_loader = prepare_loader(cfg, split='test')  # 只有一個 fold 的 (DataFrame, DataLoader)
        preds = predict(cfg, test_loader)  # test_loader(images, labels)
        preds_n = 0

        # self.add_imsizes_when_inference = [(0, 0)] -> add_imsizes_n = 0
        for add_imsizes_n, add_imsizes in enumerate(cfg.add_imsizes_when_inference):
            # self.tta = 1 (all condition)
            for tta_n in range(cfg.tta):  # 處理 TTA 次數；TTA 通常只在「推論」時使用 -> 對
                if add_imsizes_n == 0:  # here
                    suffix = ''  # 動態生成欄位名稱後綴(suffix)
                else:
                    suffix = f'multi_scale_{add_imsizes_n}_'
                if tta_n != 0:
                    suffix += f'flip_{tta_n}'
                if suffix == '':  # here
                    pred_cols = [f'pred_{c}' for c in cfg.label_features]
                else:
                    pred_cols = [f'pred_{c}_{suffix}' for c in cfg.label_features]
                test[pred_cols] = preds[preds_n]
                preds_n += 1

        test.to_csv(f'{OUTPUT_PATH}/test_fold{args.fold}.csv', index=False)  # /kaggle/working/ckpt/rsna_sagittal_level_cl_spinal_v1/test_fold0.csv
        # test.to_csv(f'{OUTPUT_PATH}/train_fold{args.fold}.csv', index=False)
        print(f'test save to {OUTPUT_PATH}/test_fold{args.fold}.csv')
        # print(f'train save to {OUTPUT_PATH}/train_fold{args.fold}.csv')


    # if cfg.predict_train:
    #     train_df, train_loader = prepare_loader(cfg, split='train')  # -> src/utils/dataloader_factory.py
    #     preds = predict(cfg, train_loader)

    #     preds_n = 0
    #     for add_imsizes_n, add_imsizes in enumerate(cfg.add_imsizes_when_inference):
    #         for tta_n in range(cfg.tta):
    #             suffix = ''
    #             if add_imsizes_n != 0:
    #                 suffix = f'multi_scale_{add_imsizes_n}_'
    #             if tta_n != 0:
    #                 suffix += f'flip_{tta_n}'
    #             if suffix == '':
    #                 pred_cols = [f'pred_{c}' for c in cfg.label_features]
    #             else:
    #                 pred_cols = [f'pred_{c}_{suffix}' for c in cfg.label_features]
    #             train_df[pred_cols] = preds[preds_n]
    #             preds_n += 1

    #     train_df.to_csv(f'{OUTPUT_PATH}/train_fold{args.fold}.csv', index=False)
    #     print(f'train predictions saved to {OUTPUT_PATH}/train_fold{args.fold}.csv')

    if cfg.predict_train:
        train_df, train_loader = prepare_loader(cfg, split='train')
        preds = predict(cfg, train_loader)

        # 統一預測欄位名稱為 pred_pred_*
        pred_cols = [f'pred_pred_{c}' for c in cfg.label_features]
        train_df[pred_cols] = preds[0]  # 注意：這假設 preds[0] 是 numpy array or tensor

        train_df.to_csv(f'{OUTPUT_PATH}/oof_train_fold{args.fold}.csv', index=False)
        print(f'train oof saved to {OUTPUT_PATH}/oof_train_fold{args.fold}.csv')

print('predict.py finish')

'''
# if cfg.predict_valid:  # 驗證集的預測；val_dataloader 定義在 data_module/classificaiton.py 中；val_loader 是以(image, label) 的資料型態回傳
#     val, val_loader = prepare_loader(cfg, split='val')  # val：通常是一個 DataFrame，包含驗證集的原始資料（例如 ID、標籤等）；val_loader：是一個資料載入器（DataLoader），可以依批次讀取驗證集的資料，方便後續推論使用。
#     preds = predict(cfg, val_loader)
#     pred_cols = [f'pred_{c}' for c in cfg.label_features]  # self.label_features = ['l1_spinal', 'l2_spinal', 'l3_spinal', 'l4_spinal', 'l5_spinal']
#     val[pred_cols] = preds[0]

# 假設 preds[0] 是一個形狀為 (N, 5) 的 numpy 陣列，其中 N 是資料筆數、5 表示五個標籤的預測值，則每一行包含這 5 個標籤的預測結果
# val['pred_l1_spinal'] 會接收到 preds[0][:, 0]（第 0 個預測值）
# val['pred_l2_spinal'] 會接收到 preds[0][:, 1]（第 1 個預測值）
'''

'''
# 驗證集(Valid):
#     用途： 用於在訓練過程中評估模型的表現，並根據其結果調整超參數（例如學習率、正則化參數等）。 
#     角色： 幫助檢測模型是否過擬合，以及作為早停（early stopping）等技術的依據。
#     來源： 通常從訓練資料中劃分出一部分，或者在交叉驗證中，根據不同 fold 來動態分割。

# 測試集(Test):
#     用途： 用來做最終的模型評估，是完全獨立於訓練和驗證的數據，確保模型的泛化能力。
#     角色： 在模型訓練和調參完成後，用測試集進行性能評估，得到最終指標。
#     來源： 通常由外部數據來源獨立收集，不參與訓練和驗證的任何過程。


# 驗證集的部分：
#   使用單一組預測結果（preds[0]）來填充 DataFrame，主要目的是評估模型在驗證集上的表現，不涉及多尺度或 TTA 的融合。
# 測試集的部分：
#   使用多尺度和 TTA 產生多組預測結果，動態生成欄位名稱後將這些結果存入 DataFrame，以便最終聚合處理，從而提升預測的穩定性與準確性。
'''

# 最後 class rsna_sagittal_cl 出來的結果會去哪邊使用呢？
# 找出來的 silce 是以什麼角度為主？(Subarticular Stenosis\Neural Foraminal Narrowing\Spinal Canal Stenosis)