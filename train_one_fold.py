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
import ssl

# 禁用 SSL 驗證
ssl._create_default_https_context = ssl._create_unverified_context

def parse_args():
    parser = argparse.ArgumentParser()
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
    if args.type == 'classification':
        from src.configs import *
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

    try:
        cfg = eval(args.config)(args.fold)
    except Exception as e:
        cfg = eval(args.config)()


    if args.gpu != 'nochange': cfg.gpu = args.gpu
    if cfg.gpu == 'big':
        devices = 8
    elif cfg.gpu == 'small':
        devices = 1
    elif cfg.gpu == 'v100':
        devices = 4
    else:
        raise
    if cfg.gpu == 'v100':
        if cfg.batch_size >= 4:
            cfg.batch_size = cfg.batch_size // 4
            cfg.grad_accumulations *= 4
        elif cfg.batch_size >= 2:
            cfg.grad_accumulations *= cfg.batch_size
            cfg.batch_size = 1

    if cfg.inference_only:
        exit()
    if cfg.train_by_all_data & (args.fold != 0):
        exit()
    cfg.fold = args.fold
    if cfg.seed is None:
        now = datetime.datetime.now()
        cfg.seed = int(now.strftime('%s'))

    RESULTS_PATH_BASE = f'results'

    if args.type == 'classification':
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
        n_gpu = torch.cuda.device_count()
        cfg.n_cpu = n_gpu * np.min([cpu_count(), cfg.batch_size])

    n_gpu = 1

    if n_gpu == 4:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    if args.type not in ['nlp', 'mlp_with_nlp', 'mlp', 'gnn']:
        if type(cfg.image_size) == int:
            cfg.image_size = (cfg.image_size, cfg.image_size)
        cfg.transform = cfg.transform(cfg.image_size)

    seed_everything(cfg.seed)
#    OUTPUT_PATH = f'{RESULTS_PATH_BASE}/{args.config}'
    OUTPUT_PATH = f'/kaggle/working/ckpt/{args.config}'
    cfg.output_path = OUTPUT_PATH
    os.system(f'mkdir -p {cfg.output_path}/val_preds/fold{args.fold}')
    logger = CSVLogger(save_dir=OUTPUT_PATH, name=f"fold_{args.fold}")

    monitor = 'val_metric'
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_PATH, filename=f"fold_{args.fold}", auto_insert_metric_name=False,
        save_top_k=cfg.save_top_k, monitor=monitor, mode='max', verbose=True, save_last=False)

    early_stop_callback = EarlyStopping(patience=cfg.early_stop_patience,
        monitor=monitor, mode='max', verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    if cfg.gpu == 'small':
        strategy = None
    else:
        strategy = 'ddp'

    # https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html
    trainer = Trainer(
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
        sync_batchnorm=cfg.sync_batchnorm,
        enable_progress_bar=False,
        resume_from_checkpoint=f'{OUTPUT_PATH}/fold_{args.fold}.ckpt' if cfg.resume else None,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        strategy=strategy,
        devices=devices if torch.cuda.is_available() else None,
        reload_dataloaders_every_n_epochs=getattr(cfg, 'reload_dataloaders_every_n_epochs', 0),
        fast_dev_run=args.debug,
    )
    model = MyLightningModule(cfg)
    datamodule = MyDataModule(cfg)

    print('start training.')
    trainer.fit(model, datamodule=datamodule)
    print(f"Training completed for fold {args.fold} in {time.time() - start_time:.2f} seconds.")  # 我加
#    os.system(f'ls {OUTPUT_PATH}/')
    os.system('ls -R /kaggle/working/ckpt/')  # 遞歸列出所有目錄
    torch.save(model.model.state_dict(), f'{OUTPUT_PATH}/last_fold{args.fold}.ckpt')
    best_model_path = checkpoint_callback.best_model_path
    best_model = model.load_from_checkpoint(cfg=cfg, checkpoint_path=best_model_path)
    torch.save(best_model.model.state_dict(), f'{OUTPUT_PATH}/fold_{args.fold}.ckpt')
    if args.fold == 3:
        cfg.train_df.to_csv(f'{OUTPUT_PATH}/train.csv', index=False)
    os.system(f'rm {OUTPUT_PATH}/fold_{args.fold}-v*.ckpt')
