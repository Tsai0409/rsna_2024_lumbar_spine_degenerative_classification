# yolo_configs.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
from pdb import set_trace as st
import copy

# kaggle input
DATA_KAGGLE_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"

# 設定環境變數
WORKING_DIR="/kaggle/working/duplicate"

class Baseline:
    def __init__(self):
        self.compe = 'rsna'
        self.batch_size = 8
        self.lr = 1e-3
        self.epochs = 500
        self.seed = 2021
        self.test_df = None
        self.image_size = 224
        self.negative_class_name = 'negative'
        self.fp16 = False
        self.predict_valid = True
        self.predict_test = False
        self.model_name= 'yolov5m'
        self.yaml = 'data/hyps/hyp.scratch-low.yaml' # hyp.finetune.yaml | hyp.finetune_objects365.yaml | hyp.scratch-high.yaml | hyp.scratch-low.yaml | hyp.scratch-med.yaml | hyp.scratch.yaml

        self.images_and_labels_dir = 'vindr_pngs' # same as image dir in path columns
        self.label_and_image_dir_name = 'vindr_pngs' # same as train file name
        self.inference_only = False
        self.evolve = None # | 300 (iter num)
        self.resume = False
        # self.create_label_and_image_dir = True
        self.create_label_and_image_dir = True
        self.patience = 50
        self.pretrained_path = '/groups/gca50041/ariyasu/yolo_weights/yolov5l.pt'
        self.sync_batchnorm = False
        self.train_all = False
        self.hub = None
        self.gpu = 'v100'
        self.make_labels = False
        self.heavy_aug = False  # yolox
        self.update_json = True
        self.no_aug_epochs = 15
        self.warmup_epochs = 5
        self.nmsthre = 0.45
        # self.absolute_path = os.getcwd()
        self.absolute_path = '/kaggle/working/duplicate'
### required columns ###
## train
# path
# fold
# image_width
# image_height
# class_id
# class_name
# box_cols

## test
# path
# image_id
# image_width
# image_height

################
class rsna_axial_all_images_left_yolox_x(Baseline):
#     def __init__(self):
    def __init__(self, fold=0):
        super().__init__()
        self.fold = fold  # 我加
        self.compe = 'rsna_2024'
        self.update_json = False
        self.no_aug_epochs = 15  # src/YOLOX/yolox/exp/yolox_base.py
        self.warmup_epochs = 5
        # self.pretrained_path = '/groups/gca50041/ariyasu/yolox_weights/yolox_x.pth'
        self.pretrained_path = '/kaggle/input/pretrain-7/yolox_x.pth'
        self.image_size = (512, 512)
        self.batch_size = 8
        self.predict_valid = True
        # self.train_df_path = 'input/train_axial_for_yolo_all_image_v1.csv'
        self.train_df_path = f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_6/train_axial_for_yolo_all_image_v1.csv'
        # self.train_df_path = f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_holdout_6/train_axial_for_yolo_all_image_v1.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.condition == 'Left Subarticular Stenosis']
        self.train_df['class_id'] = 0
        self.train_df['class_name'] = 'left'
        self.predict_test = True
        # self.epochs = 5
        self.epochs = 20
        val = self.train_df[self.train_df.fold==0]
        self.train_df['fold'] = -1
        self.train_df = pd.concat([self.train_df, val])
        
        # self.test_df_path = 'input/train_with_fold.csv'
        self.test_df_path = f'{WORKING_DIR}/csv_train/preprocess_4/train_with_fold.csv'
        # self.test_df_path = f'{WORKING_DIR}/csv_train/preprocess_holdout_4/train_with_fold_holdout_test.csv'
        self.test_df = pd.read_csv(self.test_df_path)
        self.test_df = self.test_df[self.test_df.series_description_y=='Axial T2']
        self.test_df['instance_number'] = self.test_df.path.apply(lambda x: int(x.split('___')[-1].replace('.png', '')))
        # ldf = pd.read_csv('input/axial_closest_df.csv')
        ldf = pd.read_csv(f'{WORKING_DIR}/csv_train/axial_level_estimation_2/axial_closest_df.csv')
        ldf = ldf[ldf.closest==1]
        ldf = ldf[ldf.dis<3]
        ldf['pred_level'] = ldf.level.values
        self.test_df = self.test_df.merge(ldf[['series_id', 'instance_number', 'pred_level', 'dis']], on=['series_id', 'instance_number'])

class rsna_axial_all_images_right_yolox_x(Baseline):
#     def __init__(self):
    def __init__(self, fold=0):
        super().__init__()
        self.fold = fold  # 我加
        self.compe = 'rsna_2024'
        self.update_json = False
        self.no_aug_epochs = 15
        self.warmup_epochs = 5
        # self.pretrained_path = '/groups/gca50041/ariyasu/yolox_weights/yolox_x.pth'
        self.pretrained_path = '/kaggle/input/pretrain-7/yolox_x.pth'
        self.image_size = (512, 512)
        self.batch_size = 8
        self.predict_valid = True
        # self.train_df_path = 'input/train_axial_for_yolo_all_image_v1.csv'
        # self.train_df_path = f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_6/train_axial_for_yolo_all_image_v1.csv'
        self.train_df_path = f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_holdout_6/train_axial_for_yolo_all_image_v1.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df = self.train_df[self.train_df.condition != 'Left Subarticular Stenosis']
        # self.train_df['class_id'] = 0
        self.train_df['class_id'] = 1
        self.train_df['class_name'] = 'right'
        self.predict_test = True
        self.epochs = 40
        # self.epochs = 20
        val = self.train_df[self.train_df.fold==0]
        self.train_df['fold'] = -1
        self.train_df = pd.concat([self.train_df, val])
        
        # self.test_df_path = 'input/train_with_fold.csv'
        # self.test_df_path = f'{WORKING_DIR}/csv_train/preprocess_4/train_with_fold.csv'
        self.test_df_path = f'{WORKING_DIR}/csv_train/preprocess_holdout_4/train_with_fold_holdout_test.csv'
        self.test_df = pd.read_csv(self.test_df_path)
        self.test_df = self.test_df[self.test_df.series_description_y=='Axial T2']
        self.test_df['instance_number'] = self.test_df.path.apply(lambda x: int(x.split('___')[-1].replace('.png', '')))
        # ldf = pd.read_csv('input/axial_closest_df.csv')
        ldf = pd.read_csv(f'{WORKING_DIR}/csv_train/axial_level_estimation_2/axial_closest_df.csv')
        ldf = ldf[ldf.closest==1]
        ldf = ldf[ldf.dis<3]
        ldf['pred_level'] = ldf.level.values
        self.test_df = self.test_df.merge(ldf[['series_id', 'instance_number', 'pred_level', 'dis']], on=['series_id', 'instance_number'])

class rsna_10classes_yolox_x(Baseline):
    def __init__(self):
        super().__init__()
        self.compe = 'rsna_2024'
        self.update_json = False
        self.no_aug_epochs = 15
        self.warmup_epochs = 5
        # self.pretrained_path = '/groups/gca50041/ariyasu/yolox_weights/yolox_x.pth'
        self.pretrained_path = '/kaggle/input/pretrain-7/yolox_x.pth'  # -> 在什麼樣子的資料集做 pretrain？ -> COCO
        
        self.image_size = (512, 512)
        self.batch_size = 8
        self.predict_valid = True
        # self.train_df_path = 'input/train_for_yolo_10level_v1.csv'
        # self.train_df_path = f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_6/train_for_yolo_10level_v1.csv'  # 只有 Sagittal T2/STIR
        self.train_df_path = f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_holdout_6/train_for_yolo_10level_v1.csv'
        # self.test_df_path = 'input/train_with_fold.csv'
        # self.test_df_path = f'{WORKING_DIR}/csv_train/preprocess_4/train_with_fold.csv'
        self.test_df_path = f'{WORKING_DIR}/csv_train/preprocess_holdout_4/train_with_fold_holdout.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.test_df = pd.read_csv(self.test_df_path)
        # oof = pd.read_csv(f'results/rsna_sagittal_cl/oof.csv')
        # oof = pd.read_csv(f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_6/oof.csv')
        oof = pd.read_csv(f'{WORKING_DIR}/csv_train/region_estimation_by_yolox_holdout_6/oof.csv')
        dfs = []
        for id, idf in oof.groupby('series_id'):
            idf = idf.sort_values(['x_pos', 'instance_number'])
            idf = idf.drop_duplicates('x_pos')
            ldf = idf[idf['pred_spinal']==idf['pred_spinal'].max()].iloc[:1]
            dfs.append(ldf)
        self.test_df = pd.concat(dfs)
        self.predict_test = True
        self.epochs = 40
        # self.epochs = 20
        # self.inference_only = True -> /kaggle/working/duplicate/results/rsna_10classes_yolox_x/fold0/rsna_10classes_yolox_x/best_ckpt.pth 不會產生
        self.inference_only = False 