# train_one_fold.py
import os
import shutil
from multiprocessing import cpu_count
import sys
import datetime
import time

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers.csv_logs import CSVLogger

from pdb import set_trace as st
import warnings
warnings.simplefilter('ignore')

import argparse

# 關閉 SSL 驗證
import ssl
import urllib3

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# 關閉 SSL 驗證

def parse_args():
    parser = argparse.ArgumentParser()  # 建立一個 ArgumentParser 物件
    parser.add_argument("--config", '-c', type=str, default='Test', help="config name in configs.py")  # 呼叫 configs.py
    # 如何知道是要選擇 class rsna_sagittal_level_cl_spinal_v1 及 class rsna_sagittal_level_cl_nfn_v1 -> 在 inf_sagittal_slice_1st.sh 有寫出
    parser.add_argument("--type", '-t', type=str, default='classification')
    parser.add_argument("--gpu", '-g', type=str, default='nochange')
    parser.add_argument("--debug", action='store_true', help="debug")
    parser.add_argument("--fold", '-f', type=int, default=0, help="fold num")
    return parser.parse_args()

if __name__ == "__main__":
    start = time.time()
    args = parse_args()
    print(f"Starting training for config: {args.config}, fold: {args.fold}")  # 我加

    if args.type == 'classification':  # args.type 如果沒有定義 default='classification'
        from src.configs import *  # cfg 參數的初始定義是從 configs.py 的 class Baseline 來的
    elif args.type == 'seg':
        from src.seg_configs import *
    elif args.type == 'effdet':
        from src.effdet_configs import *
    elif args.type == 'nlp':
        from src.nlp_configs import *
    elif args.type == 'mlp_with_nlp':
        from src.mlp_with_nlp_configs import *
    elif args.type == 'mlp':
        from src.mlp_configs import *
    elif args.type == 'gnn':
        from src.gnn_configs import *

    try:  # dfg 的資料來源是 configs.py
        cfg = eval(args.config)(args.fold)  # 執行一種 config\fold 建立一個 cfg 的形式，引用 configs.py 中的 class rsna_sagittal_level_cl_spinal_v1() 包含 cfg.model 等
    except Exception as e:
        cfg = eval(args.config)()

    if args.gpu != 'nochange': cfg.gpu = args.gpu
    if cfg.gpu == 'big':
        devices = 8
    elif cfg.gpu == 'small':
        devices = 1
    # self.gpu = 'v100'(all condition)
    elif cfg.gpu == 'v100':  # here
        devices = 4
    else:
        raise

    # self.gpu = 'v100'(all condition)
    if cfg.gpu == 'v100': # here
        if cfg.batch_size >= 4:
            cfg.batch_size = cfg.batch_size // 4
            cfg.grad_accumulations *= 4
        elif cfg.batch_size >= 2:
            cfg.grad_accumulations *= cfg.batch_size
            cfg.batch_size = 1

    # self.inference_only = False (all condition)
    if cfg.inference_only:
        exit()
    # self.inference_only = False (all condition)
    if cfg.train_by_all_data & (args.fold != 0):
        exit()
    cfg.fold = args.fold
    # self.seed = 2023 (all condition)
    if cfg.seed is None:  # 通常會設定 固定的 seed 值，確保每次執行程式時，隨機過程的結果是一樣的
        now = datetime.datetime.now()
        cfg.seed = int(now.strftime('%s'))

    # RESULTS_PATH_BASE = f'results'

    if args.type == 'classification':  # default='classification'
        from src.lightning.lightning_modules.classification import MyLightningModule
        from src.lightning.data_modules.classification import MyDataModule
    elif args.type == 'seg':
        from src.lightning.lightning_modules.segmentation import MyLightningModule
        from src.lightning.data_modules.segmentation import MyDataModule
    elif args.type == 'effdet':
        from src.lightning.lightning_modules.effdet import MyLightningModule
        from src.lightning.data_modules.effdet import MyDataModule
    elif args.type == 'nlp':
        from src.lightning.lightning_modules.nlp import MyLightningModule
        from src.lightning.data_modules.nlp import MyDataModule
    elif args.type == 'mlp_with_nlp':
        from src.lightning.lightning_modules.mlp_with_nlp import MyLightningModule
        from src.lightning.data_modules.mlp_with_nlp import MyDataModule
    elif args.type == 'mlp':
        from src.lightning.lightning_modules.mlp import MyLightningModule
        from src.lightning.data_modules.mlp import MyDataModule
    elif args.type == 'gnn':
        from src.lightning.lightning_modules.gnn import MyLightningModule
        from src.lightning.data_modules.gnn import MyDataModule

    if args.debug:
        cfg.epochs = 1
        cfg.n_cpu = 1
        n_gpu = 1
    else:
        n_gpu = torch.cuda.device_count()  # 獲取可用的 GPU 數量
        cfg.n_cpu = n_gpu * np.min([cpu_count(), cfg.batch_size])

    n_gpu = 1

    if n_gpu == 4:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 由 NVIDIA CUDA 控制 哪些 GPU 可以被使用

    if args.type not in ['nlp', 'mlp_with_nlp', 'mlp', 'gnn']:
        if type(cfg.image_size) == int:
            cfg.image_size = (cfg.image_size, cfg.image_size)
        cfg.transform = cfg.transform(cfg.image_size)

    seed_everything(cfg.seed)
#    OUTPUT_PATH = f'{RESULTS_PATH_BASE}/{args.config}'
    OUTPUT_PATH = f'/kaggle/working/ckpt/{args.config}'
    cfg.output_path = OUTPUT_PATH
    # os.system(f'mkdir -p {cfg.output_path}/val_preds/fold{args.fold}')
    os.system(f'mkdir -p {cfg.output_path}/fold{args.fold}')  # mkdir -p 確保建立目錄（如果已存在則不報錯）
    logger = CSVLogger(save_dir=OUTPUT_PATH, name=f"fold_{args.fold}")  # 建立一個 CSVLogger，用來記錄訓練過程中的數據
    # For example: /kaggle/working/ckpt/rsna_model/fold_0/version_0/
    # ├── metrics.csv   # 記錄 loss、accuracy 等數據
    # ├── hparams.yaml  # 記錄模型超參數

    monitor = 'val_metric'  # 只會保留 val_metric 表現最好的那個模型 → 存成 fold_0.ckpt
    checkpoint_callback = ModelCheckpoint(  # ModelCheckpoint（儲存最佳的模型權重）
        dirpath=OUTPUT_PATH, filename=f"fold_{args.fold}", auto_insert_metric_name=False,
        save_top_k=cfg.save_top_k, monitor=monitor, mode='max', verbose=True, save_last=False)

    early_stop_callback = EarlyStopping(  # EarlyStopping（提前停止機制）
        patience=cfg.early_stop_patience, monitor=monitor, mode='max', verbose=True)
    
    lr_monitor = LearningRateMonitor(  # LearningRateMonitor（學習率監控器）
        logging_interval='epoch')  

    if cfg.gpu == 'small':
        strategy = None
    else:
        strategy = 'ddp'

    # https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html
    trainer = Trainer(  # Trainer 負責管理模型訓練
        # strategy=strategy,
        # strategy=DDPStrategy(find_unused_parameters=False),  # 我加
        strategy=DDPStrategy(find_unused_parameters=True),  # 修改這裡
        max_epochs=cfg.epochs,
        gpus=n_gpu,
        accumulate_grad_batches=cfg.grad_accumulations,
        precision=16 if cfg.fp16 else 32,
        amp_backend='native',
        deterministic=False,
        auto_select_gpus=False,
        benchmark=True,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=[logger],
        # sync_batchnorm=cfg.sync_batchnorm,
        sync_batchnorm=False,  # 禁用 SyncBatchNorm
        enable_progress_bar=False,
        resume_from_checkpoint=f'{OUTPUT_PATH}/fold_{args.fold}.ckpt' if cfg.resume else None,  # self.resume = False (all condition)
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=devices if torch.cuda.is_available() else None,
        reload_dataloaders_every_n_epochs=getattr(cfg, 'reload_dataloaders_every_n_epochs', 0),
        fast_dev_run=args.debug,
    )
    model = MyLightningModule(cfg)
    datamodule = MyDataModule(cfg)
    # LightningModule：將模型、前向傳播、訓練步驟、驗證步驟等封裝到一個 LightningModule 中
    # DataModule：定義了 train_dataloader()、val_dataloader()、test_dataloader() 等方法，這些方法返回相應的 DataLoader
    # Trainer 是 Lightning 框架提供的訓練控制器，它接收一個 LightningModule 以及一個 DataModule（或直接接收 DataLoader），
    #   然後自動處理以下部分：訓練循環（迭代 DataLoader、調用 training_step）、驗證循環（調用 validation_step）、測試循環（調用 test_step）、分布式訓練、混合精度訓練、梯度累積、早停、回調函數、日誌記錄等
    # Trainer 內部是怎麼執行的？
    
    print('start training.')
    trainer.fit(model, datamodule=datamodule)  # 訓練模型
    # os.system(f'ls {OUTPUT_PATH}/')
    os.system('ls -R /kaggle/working/ckpt/')  # 遞歸列出所有目錄
    torch.save(model.model.state_dict(), f'{OUTPUT_PATH}/last_fold{args.fold}.ckpt')  # OUTPUT_PATH = f'/kaggle/working/ckpt/{args.config}';保存最後一次訓練結果的模型權重
    print(f"{OUTPUT_PATH}/last_fold{args.fold}.ckpt create completed.")  # 我加
    best_model_path = checkpoint_callback.best_model_path  # 取得最佳模型檔案「路徑」
    best_model = model.load_from_checkpoint(cfg=cfg, checkpoint_path=best_model_path)  # 取得最佳模型檔案
    torch.save(best_model.model.state_dict(), f'{OUTPUT_PATH}/fold_{args.fold}.ckpt')
    # 假設執行 10 epoch(5 epoch 時表現最好)，在 fold 0 時
    # last_fold0.ckpt 會存 epoch 10 的執行結果 (存最後的結果)
    # fold_0.ckpt 會存 epoch 5 的執行結果 (存最好的結果)

    # if args.fold == 3:
    if args.fold == 0: # 現在只要執行 fold 0.1
        cfg.train_df.to_csv(f'{OUTPUT_PATH}/train.csv', index=False)  # 把當前使用的檔案存下來
    os.system(f'rm -f {OUTPUT_PATH}/fold_{args.fold}-v*.ckpt')  # 刪除檔名像 fold_0-v1.ckpt、fold_0-v2.ckpt 的檔案

print('train_one_fold.py finish')

# 整份檔案產生：
# last_fold0.ckpt
# fold_0.ckpt
# train.csv

# 5 種不同的輸入照片？
# model 的輸入、輸出？
# 看 Inference Code 的程式 axial 的部分