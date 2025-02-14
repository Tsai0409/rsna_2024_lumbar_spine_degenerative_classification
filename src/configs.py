#configs.py
from pathlib import Path
from pprint import pprint
import timm
from src.utils.metric_learning_loss import *
from src.utils.metrics import *
from src.utils.loss import *
from src.global_objectives import AUCPRLoss
import os
import torch.nn as nn
import pandas as pd
import numpy as np
from pdb import set_trace as st
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import pytorch_lightning as pl
from types import MethodType
import warnings
warnings.simplefilter('ignore')

from src.models.resnet3d_csn import *
from src.models.uniformerv2 import *
from src.models.rsna import *
from src.models.rsna_2023_1st_models import *
from src.models.mil_3models import *
from src.models.layers import AdaptiveConcatPool2d, Flatten
from src.models.ch_mdl_dolg_efficientnet import ChMdlDolgEfficientnet, ArcFaceLossAdaptiveMargin
from src.models.rsna_multi_image import MultiLevelModel2
from src.models.backbones import *
from src.models.group_norm import convert_groupnorm
from src.models.batch_renorm import convert_batchrenorm
from src.models.multi_instance import MultiInstanceModel, MetaMIL, AttentionMILModel, MultiInstanceModelWithWataruAttention
from src.models.resnet import resnet18, resnet34, resnet101, resnet152
from src.models.nextvit import NextVitNet
from src.models.model_4channels import get_attention, get_resnet34, get_attention_inceptionv3
from src.models.vae import VAE, ResNet_VAE
from src.models.model_with_arcface import ArcMarginProduct, AddMarginProduct, ArcMarginProductSubcenter, ArcMarginProductOutCosine, ArcMarginProductSubcenterOutCosine, PudaeArcNet, WithArcface, WhalePrev1stModel, Guie2
from src.models.with_meta_models import WithMetaModel

from src.utils.augmentations.strong_aug import *
from src.utils.augmentations.augmentation import *
from src.utils.augmentations.policy_transform import policy_transform
from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error, average_precision_score, accuracy_score
import numpy as np
from scipy.special import softmax
def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
WORKING_DIR="/kaggle/working/duplicate"
################

class Baseline:
    def __init__(self):
        self.memo = ''
        # self.gpu = 'small'
        self.gpu = 'v100'
        self.compe = 'rsna'
        self.batch_size = 16
        self.grad_accumulations = 1
        self.lr = 0.0001
        self.epochs = 20
        self.resume = False
        self.seed = 2023
        self.tta = 1
        self.model_name = 'convnext_small.fb_in22k_ft_in1k_384'
        # self.model_name = 'resnet50'
        self.num_classes = 1
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        # self.criterion = torch.nn.BCELoss()
        # self.transform = medical_v1
        self.transform = kuma_aug
        self.image_size = 384
        self.label_features = ['target']
        self.metric = roc_auc_score # AUC().torch # MultiAP().torch # MultiAUC().torch
        self.fp16 = True
        # self.optimizer = 'adam'
        self.optimizer = 'adamw'
        self.scheduler = 'CosineAnnealingWarmRestarts'
        self.eta_min = 5e-7
        self.train_by_all_data = False
        self.early_stop_patience = 1000
        self.inference = False
        self.predict_valid = True
        self.predict_test = False
        self.logit_to = None
        self.pretrained_path = None
        self.sync_batchnorm = True
        # self.sync_batchnorm = False
        self.warmup_epochs = -1
        self.finetune_transform = base_aug_v1
        self.mixup = False
        self.arcface = False
        self.box_crop = None
        self.predicted_mask_crop = None
        self.pad_square = False
        self.resume_epoch = 0
        self.t_max=30
        self.save_top_k = 1
        self.meta_cols = []
        self.output_features = False
        self.force_use_model_path_config_when_inf = None
        self.reset_classifier_when_inf = False
        self.upsample = None
        self.in_chans = 3
        self.add_imsizes_when_inference = [(0, 0)]
        self.inf_fp16 = False
        self.distill = False
        self.reload_dataloaders_every_n_epochs = 0
        self.tranform_dataset_version = None
        self.no_trained_model_when_inf = False
        self.normalize_horiz_orientation = False
        self.upsample_batch_pos_n = None
        self.cut_200 = False
        self.affine_for_gbr = False
        self.half_dark = False
        self.crop_by_left_right_line_text = False
        self.use_wandb = True
        self.use_last_ckpt_when_inference = True
        self.inference_only = False
        self.valid_df = None
        self.valid_df_path = None
        self.ema = False
        self.awp = False
        self.save_every_epoch_val_preds = False

class rsna_v1(Baseline):
    def __init__(self):
        super().__init__()
        self.compe = 'rsna_2024'
        self.predict_valid = True
        self.predict_test = False
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.transform = medical_v3
        self.batch_size = 8
        self.lr = 1e-5
        self.grad_accumulations = 2
        self.p_rand_order_v1 = 0

class rsna_sagittal_level_cl_spinal_v1(rsna_v1):
    def __init__(self, fold=0):
        super().__init__()
        self.fold = fold  # 如果你要在這裡使用 fold，就給它存起來
#        self.train_df_path = 'input/train_for_sagittal_level_cl_v1_for_train_spinal_only.csv'
        self.train_df_path = f'{WORKING_DIR}/csv_train/sagittal_slice_estimation_5/train_for_sagittal_level_cl_v1_for_train_spinal_only.csv'
        print("I'm reading from path:", self.train_df_path)
        self.train_df = pd.read_csv(self.train_df_path)
        self.label_features = ['l1_spinal', 'l2_spinal', 'l3_spinal', 'l4_spinal', 'l5_spinal']
        self.num_classes = len(self.label_features)
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.image_size = 256
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        self.metric = MultiAUC(label_features=self.label_features).torch
        self.memo = ''
        self.batch_size = 16
        self.grad_accumulations = 1
        self.crop_by_xy = False
        self.rsna_2024_multi_image = False
        self.rsna_random_sample = False
        self.lr = 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 15
        self.box_crop = None
#        self.test_df = pd.read_csv('input/sagittal_df.csv')
        self.test_df = pd.read_csv(f'{WORKING_DIR}/csv_train/dcm_to_png_3/sagittal_df.csv')
        self.predict_test = True

class rsna_sagittal_level_cl_nfn_v1(rsna_v1):
    def __init__(self, fold=0):
        super().__init__()
        self.fold = fold  # 如果你要在這裡使用 fold，就給它存起來
#        self.train_df_path = 'input/train_for_sagittal_level_cl_v1_for_train_nfn_only.csv'
        self.train_df_path = f'{WORKING_DIR}/csv_train/sagittal_slice_estimation_5/train_for_sagittal_level_cl_v1_for_train_nfn_only.csv'
        print("I'm reading from path:", self.train_df_path)
        self.train_df = pd.read_csv(self.train_df_path)
        self.label_features = ['l1_right_neural', 'l2_right_neural', 'l3_right_neural', 'l4_right_neural', 'l5_right_neural', 'l1_left_neural', 'l2_left_neural', 'l3_left_neural', 'l4_left_neural', 'l5_left_neural']
        self.num_classes = len(self.label_features)
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.image_size = 256
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        self.metric = MultiAUC(label_features=self.label_features).torch
        self.memo = ''
        self.batch_size = 16
        self.grad_accumulations = 1
        self.crop_by_xy = False
        self.rsna_2024_multi_image = False
        self.rsna_random_sample = False
        self.lr = 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 15
        self.box_crop = None
#        self.test_df = pd.read_csv('input/sagittal_df.csv')
        self.test_df = pd.read_csv(f'{WORKING_DIR}/csv_train/dcm_to_png_3/sagittal_df.csv')
        self.predict_test = True

class rsna_sagittal_cl(rsna_v1):
    def __init__(self, fold):
        super().__init__()
#        self.train_df_path = f'input/train_for_sagittal_level_cl_v1_for_train_spinal_nfn_fold{fold}.csv'
        self.train_df_path = f'{WORKING_DIR}/train_for_sagittal_level_cl_v1_for_train_spinal_nfn_fold{fold}.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.label_features = ['l1_spinal', 'l2_spinal', 'l3_spinal', 'l4_spinal', 'l5_spinal', 'l1_right_neural', 'l2_right_neural', 'l3_right_neural', 'l4_right_neural', 'l5_right_neural', 'l1_left_neural', 'l2_left_neural', 'l3_left_neural', 'l4_left_neural', 'l5_left_neural']
        self.label_features = ['pred_'+c for c in self.label_features]
        self.num_classes = len(self.label_features)
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.image_size = 256
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        self.metric = None
        self.memo = ''
        self.batch_size = 16
        self.grad_accumulations = 1
        self.crop_by_xy = False
        self.rsna_2024_multi_image = False
        self.rsna_random_sample = False
        self.lr = 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 15
        self.box_crop = None
        self.predict_test = False

class rsna_axial_spinal_crop_base(rsna_v1):
    def __init__(self):
        super().__init__()
#        self.train_df_path = 'input/axial_classification.csv'
        self.train_df_path = '/kaggle/working/rsna_2024_lumbar_spine_degenerative_classification-main/axial_classification.csv'
        self.train_df = pd.read_csv(self.train_df_path)

        cols = []
        label_features = [
            'spinal_canal_stenosis',
        ]
        for col in label_features:
            cols.append(f'{col}_normal')
            cols.append(f'{col}_moderate')
            cols.append(f'{col}_severe')

        self.label_features = cols
        self.num_classes = len(self.label_features)
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.image_size = 384
        self.drop_rate = 0.0
        self.drop_path_rate = 0.0
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes,
            drop_rate=self.drop_rate, drop_path_rate=self.drop_path_rate)
        self.metric = None
        self.memo = ''
        self.batch_size = 8
        self.grad_accumulations = 2
        self.crop_by_xy = False
        self.rsna_2024_multi_image = False
        self.rsna_random_sample = False
        self.lr = 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 7
        self.transform = medical_v3
        self.box_crop = True
        self.box_crop_x_ratio = 2
        self.box_crop_y_ratio = 6

class rsna_axial_spinal_dis3_crop_x05_y6(rsna_axial_spinal_crop_base):
    def __init__(self):
        super().__init__()
        self.box_crop_x_ratio = 0.5
        self.box_crop_y_ratio = 6

class rsna_axial_spinal_dis3_crop_x1_y2(rsna_axial_spinal_crop_base):
    def __init__(self):
        super().__init__()
        self.box_crop_x_ratio = 1
        self.box_crop_y_ratio = 2
        
class rsna_axial_spinal_dis3_crop_x05_y6_reduce_noise(rsna_axial_spinal_dis3_crop_x05_y6):
    def __init__(self):
        super().__init__()
        self.train_df['level'] = self.train_df.pred_level.map({
            1: 'l1_l2',
            2: 'l2_l3',
            3: 'l3_l4',
            4: 'l4_l5',
            5: 'l5_s1',
        })
        noise_df = pd.read_csv(f'results/noisy_target_level_1016.csv')
        noise_df = noise_df[noise_df.target=='spinal_canal_stenosis']
        self.train_df['study_level'] = self.train_df.study_id.astype(str) + '_' + self.train_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.train_df = self.train_df[~self.train_df.study_level.isin(noise_df.study_level)]

class rsna_axial_spinal_dis3_crop_x1_y2_reduce_noise(rsna_axial_spinal_dis3_crop_x1_y2):
    def __init__(self):
        super().__init__()
        self.train_df['level'] = self.train_df.pred_level.map({
            1: 'l1_l2',
            2: 'l2_l3',
            3: 'l3_l4',
            4: 'l4_l5',
            5: 'l5_s1',
        })
        noise_df = pd.read_csv(f'results/noisy_target_level_1016.csv')
        noise_df = noise_df[noise_df.target=='spinal_canal_stenosis']
        self.train_df['study_level'] = self.train_df.study_id.astype(str) + '_' + self.train_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.train_df = self.train_df[~self.train_df.study_level.isin(noise_df.study_level)]

class rsna_axial_ss_nfn_crop_base(rsna_v1):
    def __init__(self):
        super().__init__()
        cols = []
        label_features = [
            'neural_foraminal_narrowing',
            'subarticular_stenosis',
        ]
        for col in label_features:
            cols.append(f'{col}_normal')
            cols.append(f'{col}_moderate')
            cols.append(f'{col}_severe')

        self.label_features = cols
        self.num_classes = len(self.label_features)
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.image_size = 384
        self.drop_rate = 0.0
        self.drop_path_rate = 0.0
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes,
            drop_rate=self.drop_rate, drop_path_rate=self.drop_path_rate)
        self.metric = None
        self.memo = ''
        self.batch_size = 8
        self.grad_accumulations = 2
        self.crop_by_xy = False
        self.rsna_2024_multi_image = False
        self.rsna_random_sample = False
        self.lr = 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 7
        self.transform = medical_v4
        self.box_crop = True
        self.box_crop_x_ratio = 0
        self.box_crop_y_ratio = 6


class rsna_axial_ss_nfn_x2_y2_center_pad0(rsna_axial_ss_nfn_crop_base):
    def __init__(self):
        super().__init__()
        image_width_ratio = 2
        self.box_crop_y_ratio = 2
        center_pad_ratio = 0

#        self.train_df_path = 'input/axial_classification.csv'
        self.train_df_path = '/kaggle/working/rsna_2024_lumbar_spine_degenerative_classification-main/axial_classification.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['x_min'] = (self.train_df.x_max + self.train_df.x_min)/2
        del self.train_df['x_max']
        self.train_df['left_right'] = 'left'
        cols = [
            'left_neural_foraminal_narrowing_normal',           
            'left_neural_foraminal_narrowing_moderate',
            'left_neural_foraminal_narrowing_severe',
            'left_subarticular_stenosis_normal',           
            'left_subarticular_stenosis_moderate',
            'left_subarticular_stenosis_severe',
        ]
        for c in cols:
            self.train_df[c.replace('left_', '')] = self.train_df[c].values
        self.train_df['x_max'] = self.train_df['x_min'] + self.train_df['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            self.train_df['x_min'] = self.train_df['x_min'] - self.train_df['image_width']/center_pad_ratio

        train_df_right = pd.read_csv(self.train_df_path)
        train_df_right['x_max'] = (train_df_right.x_max + train_df_right.x_min)/2
        del train_df_right['x_min']
        train_df_right['left_right'] = 'right'
        cols = [
            'right_neural_foraminal_narrowing_normal',           
            'right_neural_foraminal_narrowing_moderate',
            'right_neural_foraminal_narrowing_severe',
            'right_subarticular_stenosis_normal',           
            'right_subarticular_stenosis_moderate',
            'right_subarticular_stenosis_severe',
        ]
        for c in cols:
            train_df_right[c.replace('right_', '')] = train_df_right[c].values
        train_df_right['x_min'] = train_df_right['x_max'] - train_df_right['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            train_df_right['x_max'] = train_df_right['x_max'] + train_df_right['image_width']/center_pad_ratio

        self.train_df = pd.concat([self.train_df, train_df_right])

class rsna_axial_ss_nfn_x2_y6_center_pad0(rsna_axial_ss_nfn_crop_base):
    def __init__(self):
        super().__init__()
        image_width_ratio = 2
        self.box_crop_y_ratio = 6
        center_pad_ratio = 0

#        self.train_df_path = 'input/axial_classification.csv'
        self.train_df_path = '/kaggle/working/rsna_2024_lumbar_spine_degenerative_classification-main/axial_classification.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['x_min'] = (self.train_df.x_max + self.train_df.x_min)/2
        del self.train_df['x_max']
        self.train_df['left_right'] = 'left'
        cols = [
            'left_neural_foraminal_narrowing_normal',           
            'left_neural_foraminal_narrowing_moderate',
            'left_neural_foraminal_narrowing_severe',
            'left_subarticular_stenosis_normal',           
            'left_subarticular_stenosis_moderate',
            'left_subarticular_stenosis_severe',
        ]
        for c in cols:
            self.train_df[c.replace('left_', '')] = self.train_df[c].values
        self.train_df['x_max'] = self.train_df['x_min'] + self.train_df['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            self.train_df['x_min'] = self.train_df['x_min'] - self.train_df['image_width']/center_pad_ratio


        train_df_right = pd.read_csv(self.train_df_path)
        train_df_right['x_max'] = (train_df_right.x_max + train_df_right.x_min)/2
        del train_df_right['x_min']
        train_df_right['left_right'] = 'right'
        cols = [
            'right_neural_foraminal_narrowing_normal',           
            'right_neural_foraminal_narrowing_moderate',
            'right_neural_foraminal_narrowing_severe',
            'right_subarticular_stenosis_normal',           
            'right_subarticular_stenosis_moderate',
            'right_subarticular_stenosis_severe',
        ]
        for c in cols:
            train_df_right[c.replace('right_', '')] = train_df_right[c].values
        train_df_right['x_min'] = train_df_right['x_max'] - train_df_right['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            train_df_right['x_max'] = train_df_right['x_max'] + train_df_right['image_width']/center_pad_ratio

        self.train_df = pd.concat([self.train_df, train_df_right])
class rsna_axial_ss_nfn_x2_y8_center_pad10(rsna_axial_ss_nfn_crop_base):
    def __init__(self):
        super().__init__()
        image_width_ratio = 2
        self.box_crop_y_ratio = 8
        center_pad_ratio = 10

#        self.train_df_path = 'input/axial_classification.csv'
        self.train_df_path = '/kaggle/working/rsna_2024_lumbar_spine_degenerative_classification-main/axial_classification.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['x_min'] = (self.train_df.x_max + self.train_df.x_min)/2
        del self.train_df['x_max']
        self.train_df['left_right'] = 'left'
        cols = [
            'left_neural_foraminal_narrowing_normal',           
            'left_neural_foraminal_narrowing_moderate',
            'left_neural_foraminal_narrowing_severe',
            'left_subarticular_stenosis_normal',           
            'left_subarticular_stenosis_moderate',
            'left_subarticular_stenosis_severe',
        ]
        for c in cols:
            self.train_df[c.replace('left_', '')] = self.train_df[c].values
        self.train_df['x_max'] = self.train_df['x_min'] + self.train_df['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            self.train_df['x_min'] = self.train_df['x_min'] - self.train_df['image_width']/center_pad_ratio


        train_df_right = pd.read_csv(self.train_df_path)
        train_df_right['x_max'] = (train_df_right.x_max + train_df_right.x_min)/2
        del train_df_right['x_min']
        train_df_right['left_right'] = 'right'
        cols = [
            'right_neural_foraminal_narrowing_normal',           
            'right_neural_foraminal_narrowing_moderate',
            'right_neural_foraminal_narrowing_severe',
            'right_subarticular_stenosis_normal',           
            'right_subarticular_stenosis_moderate',
            'right_subarticular_stenosis_severe',
        ]
        for c in cols:
            train_df_right[c.replace('right_', '')] = train_df_right[c].values
        train_df_right['x_min'] = train_df_right['x_max'] - train_df_right['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            train_df_right['x_max'] = train_df_right['x_max'] + train_df_right['image_width']/center_pad_ratio

        self.train_df = pd.concat([self.train_df, train_df_right])

class rsna_axial_ss_nfn_x2_y2_center_pad0_reduce_noise(rsna_axial_ss_nfn_crop_base):
    def __init__(self):
        super().__init__()
        image_width_ratio = 2
        self.box_crop_y_ratio = 2
        center_pad_ratio = 0

#        self.train_df_path = 'input/axial_classification.csv'
        self.train_df_path = '/kaggle/working/rsna_2024_lumbar_spine_degenerative_classification-main/axial_classification.csv'
        self.train_df = pd.read_csv(self.train_df_path)

        self.train_df['level'] = self.train_df.pred_level.map({
            1: 'l1_l2',
            2: 'l2_l3',
            3: 'l3_l4',
            4: 'l4_l5',
            5: 'l5_s1',
        })
        noise_df = pd.read_csv(f'results/noisy_target_level_th09.csv')
        noise_df = noise_df[(noise_df.target == 'left_neural_foraminal_narrowing') | (noise_df.target == 'left_subarticular_stenosis')]
        self.train_df['study_level'] = self.train_df.study_id.astype(str) + '_' + self.train_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.train_df = self.train_df[~self.train_df.study_level.isin(noise_df.study_level)]

        self.train_df['x_min'] = (self.train_df.x_max + self.train_df.x_min)/2
        del self.train_df['x_max']
        self.train_df['left_right'] = 'left'
        cols = [
            'left_neural_foraminal_narrowing_normal',           
            'left_neural_foraminal_narrowing_moderate',
            'left_neural_foraminal_narrowing_severe',
            'left_subarticular_stenosis_normal',           
            'left_subarticular_stenosis_moderate',
            'left_subarticular_stenosis_severe',
        ]
        for c in cols:
            self.train_df[c.replace('left_', '')] = self.train_df[c].values
        self.train_df['x_max'] = self.train_df['x_min'] + self.train_df['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            self.train_df['x_min'] = self.train_df['x_min'] - self.train_df['image_width']/center_pad_ratio

        train_df_right = pd.read_csv(self.train_df_path)
        train_df_right['level'] = train_df_right.pred_level.map({
            1: 'l1_l2',
            2: 'l2_l3',
            3: 'l3_l4',
            4: 'l4_l5',
            5: 'l5_s1',
        })
        noise_df = pd.read_csv(f'results/noisy_target_level_th09.csv')
        noise_df = noise_df[(noise_df.target == 'right_neural_foraminal_narrowing') | (noise_df.target == 'right_subarticular_stenosis')]
        train_df_right['study_level'] = train_df_right.study_id.astype(str) + '_' + train_df_right.level.apply(lambda x: x.replace('/', '_').lower())
        train_df_right = train_df_right[~train_df_right.study_level.isin(noise_df.study_level)]

        train_df_right['x_max'] = (train_df_right.x_max + train_df_right.x_min)/2
        del train_df_right['x_min']
        train_df_right['left_right'] = 'right'
        cols = [
            'right_neural_foraminal_narrowing_normal',           
            'right_neural_foraminal_narrowing_moderate',
            'right_neural_foraminal_narrowing_severe',
            'right_subarticular_stenosis_normal',           
            'right_subarticular_stenosis_moderate',
            'right_subarticular_stenosis_severe',
        ]
        for c in cols:
            train_df_right[c.replace('right_', '')] = train_df_right[c].values
        train_df_right['x_min'] = train_df_right['x_max'] - train_df_right['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            train_df_right['x_max'] = train_df_right['x_max'] + train_df_right['image_width']/center_pad_ratio

        self.train_df = pd.concat([self.train_df, train_df_right])

class rsna_axial_ss_nfn_x2_y6_center_pad0_reduce_noise(rsna_axial_ss_nfn_crop_base):
    def __init__(self):
        super().__init__()
        image_width_ratio = 2
        self.box_crop_y_ratio = 6
        center_pad_ratio = 0

#        self.train_df_path = 'input/axial_classification.csv'
        self.train_df_path = '/kaggle/working/rsna_2024_lumbar_spine_degenerative_classification-main/axial_classification.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['level'] = self.train_df.pred_level.map({
            1: 'l1_l2',
            2: 'l2_l3',
            3: 'l3_l4',
            4: 'l4_l5',
            5: 'l5_s1',
        })
        noise_df = pd.read_csv(f'results/noisy_target_level_th09.csv')
        noise_df = noise_df[(noise_df.target == 'left_neural_foraminal_narrowing') | (noise_df.target == 'left_subarticular_stenosis')]
        self.train_df['study_level'] = self.train_df.study_id.astype(str) + '_' + self.train_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.train_df = self.train_df[~self.train_df.study_level.isin(noise_df.study_level)]        
        self.train_df['x_min'] = (self.train_df.x_max + self.train_df.x_min)/2
        del self.train_df['x_max']
        self.train_df['left_right'] = 'left'
        cols = [
            'left_neural_foraminal_narrowing_normal',           
            'left_neural_foraminal_narrowing_moderate',
            'left_neural_foraminal_narrowing_severe',
            'left_subarticular_stenosis_normal',           
            'left_subarticular_stenosis_moderate',
            'left_subarticular_stenosis_severe',
        ]
        for c in cols:
            self.train_df[c.replace('left_', '')] = self.train_df[c].values
        self.train_df['x_max'] = self.train_df['x_min'] + self.train_df['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            self.train_df['x_min'] = self.train_df['x_min'] - self.train_df['image_width']/center_pad_ratio


        train_df_right = pd.read_csv(self.train_df_path)
        train_df_right['level'] = train_df_right.pred_level.map({
            1: 'l1_l2',
            2: 'l2_l3',
            3: 'l3_l4',
            4: 'l4_l5',
            5: 'l5_s1',
        })
        noise_df = pd.read_csv(f'results/noisy_target_level_th09.csv')
        noise_df = noise_df[(noise_df.target == 'right_neural_foraminal_narrowing') | (noise_df.target == 'right_subarticular_stenosis')]
        train_df_right['study_level'] = train_df_right.study_id.astype(str) + '_' + train_df_right.level.apply(lambda x: x.replace('/', '_').lower())
        train_df_right = train_df_right[~train_df_right.study_level.isin(noise_df.study_level)]        
        train_df_right['x_max'] = (train_df_right.x_max + train_df_right.x_min)/2
        del train_df_right['x_min']
        train_df_right['left_right'] = 'right'
        cols = [
            'right_neural_foraminal_narrowing_normal',           
            'right_neural_foraminal_narrowing_moderate',
            'right_neural_foraminal_narrowing_severe',
            'right_subarticular_stenosis_normal',           
            'right_subarticular_stenosis_moderate',
            'right_subarticular_stenosis_severe',
        ]
        for c in cols:
            train_df_right[c.replace('right_', '')] = train_df_right[c].values
        train_df_right['x_min'] = train_df_right['x_max'] - train_df_right['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            train_df_right['x_max'] = train_df_right['x_max'] + train_df_right['image_width']/center_pad_ratio

        self.train_df = pd.concat([self.train_df, train_df_right])
class rsna_axial_ss_nfn_x2_y8_center_pad10_reduce_noise(rsna_axial_ss_nfn_crop_base):
    def __init__(self):
        super().__init__()
        image_width_ratio = 2
        self.box_crop_y_ratio = 8
        center_pad_ratio = 10

#        self.train_df_path = 'input/axial_classification.csv'
        self.train_df_path = '/kaggle/working/rsna_2024_lumbar_spine_degenerative_classification-main/axial_classification.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['level'] = self.train_df.pred_level.map({
            1: 'l1_l2',
            2: 'l2_l3',
            3: 'l3_l4',
            4: 'l4_l5',
            5: 'l5_s1',
        })
        noise_df = pd.read_csv(f'results/noisy_target_level_th09.csv')
        noise_df = noise_df[(noise_df.target == 'left_neural_foraminal_narrowing') | (noise_df.target == 'left_subarticular_stenosis')]
        self.train_df['study_level'] = self.train_df.study_id.astype(str) + '_' + self.train_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.train_df = self.train_df[~self.train_df.study_level.isin(noise_df.study_level)]        
        self.train_df['x_min'] = (self.train_df.x_max + self.train_df.x_min)/2
        del self.train_df['x_max']
        self.train_df['left_right'] = 'left'
        cols = [
            'left_neural_foraminal_narrowing_normal',           
            'left_neural_foraminal_narrowing_moderate',
            'left_neural_foraminal_narrowing_severe',
            'left_subarticular_stenosis_normal',           
            'left_subarticular_stenosis_moderate',
            'left_subarticular_stenosis_severe',
        ]
        for c in cols:
            self.train_df[c.replace('left_', '')] = self.train_df[c].values
        self.train_df['x_max'] = self.train_df['x_min'] + self.train_df['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            self.train_df['x_min'] = self.train_df['x_min'] - self.train_df['image_width']/center_pad_ratio

        train_df_right = pd.read_csv(self.train_df_path)
        train_df_right['level'] = train_df_right.pred_level.map({
            1: 'l1_l2',
            2: 'l2_l3',
            3: 'l3_l4',
            4: 'l4_l5',
            5: 'l5_s1',
        })
        noise_df = pd.read_csv(f'results/noisy_target_level_th09.csv')
        noise_df = noise_df[(noise_df.target == 'right_neural_foraminal_narrowing') | (noise_df.target == 'right_subarticular_stenosis')]
        train_df_right['study_level'] = train_df_right.study_id.astype(str) + '_' + train_df_right.level.apply(lambda x: x.replace('/', '_').lower())
        train_df_right = train_df_right[~train_df_right.study_level.isin(noise_df.study_level)]        
        train_df_right['x_max'] = (train_df_right.x_max + train_df_right.x_min)/2
        del train_df_right['x_min']
        train_df_right['left_right'] = 'right'
        cols = [
            'right_neural_foraminal_narrowing_normal',           
            'right_neural_foraminal_narrowing_moderate',
            'right_neural_foraminal_narrowing_severe',
            'right_subarticular_stenosis_normal',           
            'right_subarticular_stenosis_moderate',
            'right_subarticular_stenosis_severe',
        ]
        for c in cols:
            train_df_right[c.replace('right_', '')] = train_df_right[c].values
        train_df_right['x_min'] = train_df_right['x_max'] - train_df_right['image_width']/image_width_ratio
        if center_pad_ratio!=0:
            train_df_right['x_max'] = train_df_right['x_max'] + train_df_right['image_width']/center_pad_ratio

        self.train_df = pd.concat([self.train_df, train_df_right])


class rsna_saggital_spinal_crop_base(rsna_v1):
    def __init__(self):
        super().__init__()
        self.train_df_path = 'input/sagittal_spinal_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        dfs = []
        col = 'spinal_canal_stenosis'
        for level, idf in self.train_df.groupby('level'):
            idf[f'{col}_normal'] = 0
            idf[f'{col}_moderate'] = 0
            idf[f'{col}_severe'] = 0
            idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Normal/Mild', f'{col}_normal'] = 1
            idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Moderate', f'{col}_moderate'] = 1
            idf.loc[idf[col+'_'+level.replace('/', '_').lower()]=='Severe', f'{col}_severe'] = 1
            idf = idf[~idf[col+'_'+level.replace('/', '_').lower()].isnull()]
            dfs.append(idf)
        self.train_df = pd.concat(dfs)            
        self.label_features = [
            'spinal_canal_stenosis_normal',
            'spinal_canal_stenosis_moderate',
            'spinal_canal_stenosis_severe',
        ]
        self.num_classes = len(self.label_features)
        self.model_name = 'convnext_small.in12k_ft_in1k_384'
        self.drop_rate = 0.0
        self.drop_path_rate = 0.0
        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=1,
            drop_rate=self.drop_rate, drop_path_rate=self.drop_path_rate)
        self.model = RSNA2ndModel(base_model=base_model,
            num_classes=len(self.label_features), pool='avg', swin=False)
        self.metric = None
        self.memo = ''
        self.lr = 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 4
        self.image_size = 128
        self.batch_size = 16
        self.grad_accumulations = 1
        self.use_sagittal_mil_dataset = True
        self.box_crop = True
        self.box_crop_x_ratio = 1
        self.box_crop_y_ratio = 0.5
class rsna_saggital_mil_spinal_crop_x03_y05(rsna_saggital_spinal_crop_base):
    def __init__(self):
        super().__init__()
        self.box_crop_x_ratio = 0.3
        self.box_crop_y_ratio = 0.5

class rsna_saggital_mil_spinal_crop_x03_y07(rsna_saggital_spinal_crop_base):
    def __init__(self):
        super().__init__()
        self.box_crop_x_ratio = 0.3
        self.box_crop_y_ratio = 0.7

class rsna_saggital_mil_spinal_crop_x03_y05_reduce_noise(rsna_saggital_mil_spinal_crop_x03_y05):
    def __init__(self):
        super().__init__()
        noise_df = pd.read_csv(f'results/noisy_target_level_1016.csv')
        noise_df = noise_df[noise_df.target=='spinal_canal_stenosis']
        self.train_df['study_level'] = self.train_df.study_id.astype(str) + '_' + self.train_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.train_df = self.train_df[~self.train_df.study_level.isin(noise_df.study_level)]

class rsna_saggital_mil_spinal_crop_x03_y07_reduce_noise(rsna_saggital_mil_spinal_crop_x03_y07):
    def __init__(self):
        super().__init__()
        noise_df = pd.read_csv(f'results/noisy_target_level_1016.csv')
        noise_df = noise_df[noise_df.target=='spinal_canal_stenosis']
        self.train_df['study_level'] = self.train_df.study_id.astype(str) + '_' + self.train_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.train_df = self.train_df[~self.train_df.study_level.isin(noise_df.study_level)]

class rsna_saggital_mil_ss_crop_base(rsna_v1):
    def __init__(self):
        super().__init__()
        self.train_df_path = 'input/sagittal_right_ss_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['left_right'] = 'right'
        self.train_df['subarticular_stenosis_normal'] = self.train_df['right_subarticular_stenosis_normal'].values
        self.train_df['subarticular_stenosis_moderate'] = self.train_df['right_subarticular_stenosis_moderate'].values
        self.train_df['subarticular_stenosis_severe'] = self.train_df['right_subarticular_stenosis_severe'].values
        self.add_df_path = 'input/sagittal_left_ss_range2_rolling5.csv'
        self.add_df = pd.read_csv(self.add_df_path)
        self.add_df['left_right'] = 'left'
        self.add_df['subarticular_stenosis_normal'] = self.add_df['left_subarticular_stenosis_normal'].values
        self.add_df['subarticular_stenosis_moderate'] = self.add_df['left_subarticular_stenosis_moderate'].values
        self.add_df['subarticular_stenosis_severe'] = self.add_df['left_subarticular_stenosis_severe'].values
        self.train_df = pd.concat([self.train_df, self.add_df])
        self.label_features = [
            'subarticular_stenosis_normal',
            'subarticular_stenosis_moderate',
            'subarticular_stenosis_severe',
        ]
        l = len(self.train_df)
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
        print(l, len(self.train_df))

        self.num_classes = len(self.label_features)
        self.model_name = 'convnext_small.in12k_ft_in1k_384'

        # self.metric = MultiAUC(label_features=self.label_features).torch
        self.metric = None
        self.memo = ''
        self.lr = 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 6
        self.batch_size = 16
        self.grad_accumulations = 1
        self.use_sagittal_mil_dataset = True
        self.drop_rate = 0.0
        self.drop_path_rate = 0.0

        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=1,
            drop_rate=self.drop_rate, drop_path_rate=self.drop_path_rate)
        self.model = RSNA2ndModel(base_model=base_model,
            num_classes=len(self.label_features), pool='avg', swin=False)
        self.box_crop = True
        self.box_crop_x_ratio = 0.4
        self.box_crop_y_ratio = 0.2

class rsna_saggital_mil_ss_crop_x03_y05_96(rsna_saggital_mil_ss_crop_base):
    def __init__(self):
        super().__init__()
        self.box_crop_x_ratio = 0.3
        self.box_crop_y_ratio = 0.5
        self.image_size = 96
class rsna_saggital_mil_ss_crop_x03_y07_96(rsna_saggital_mil_ss_crop_base):
    def __init__(self):
        super().__init__()
        self.box_crop_x_ratio = 0.3
        self.box_crop_y_ratio = 0.7
        self.image_size = 96
class rsna_saggital_mil_ss_crop_x03_y2_96(rsna_saggital_mil_ss_crop_base):
    def __init__(self):
        super().__init__()
        self.box_crop_x_ratio = 0.3
        self.box_crop_y_ratio = 2
        self.image_size = 96
class rsna_saggital_mil_ss_crop_x1_y07_96(rsna_saggital_mil_ss_crop_base):
    def __init__(self):
        super().__init__()
        self.box_crop_x_ratio = 1
        self.box_crop_y_ratio = 0.7
        self.image_size = 96

class rsna_saggital_mil_ss_crop_x03_y05_96_reduce_noise(rsna_saggital_mil_ss_crop_x03_y05_96):
    def __init__(self):
        super().__init__()
        self.train_df_path = 'input/sagittal_right_ss_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['left_right'] = 'right'
        self.train_df['subarticular_stenosis_normal'] = self.train_df['right_subarticular_stenosis_normal'].values
        self.train_df['subarticular_stenosis_moderate'] = self.train_df['right_subarticular_stenosis_moderate'].values
        self.train_df['subarticular_stenosis_severe'] = self.train_df['right_subarticular_stenosis_severe'].values

        noise_df = pd.read_csv(f'results/noisy_target_level_th08.csv')
        noise_df = noise_df[noise_df.target == 'right_subarticular_stenosis']
        self.train_df['study_level'] = self.train_df.study_id.astype(str) + '_' + self.train_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.train_df = self.train_df[~self.train_df.study_level.isin(noise_df.study_level)]

        self.add_df_path = 'input/sagittal_left_ss_range2_rolling5.csv'
        self.add_df = pd.read_csv(self.add_df_path)
        self.add_df['left_right'] = 'left'
        self.add_df['subarticular_stenosis_normal'] = self.add_df['left_subarticular_stenosis_normal'].values
        self.add_df['subarticular_stenosis_moderate'] = self.add_df['left_subarticular_stenosis_moderate'].values
        self.add_df['subarticular_stenosis_severe'] = self.add_df['left_subarticular_stenosis_severe'].values

        noise_df = pd.read_csv(f'results/noisy_target_level_th08.csv')
        noise_df = noise_df[noise_df.target == 'left_subarticular_stenosis']
        self.add_df['study_level'] = self.add_df.study_id.astype(str) + '_' + self.add_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.add_df = self.add_df[~self.add_df.study_level.isin(noise_df.study_level)]

        self.train_df = pd.concat([self.train_df, self.add_df])
        self.label_features = [
            'subarticular_stenosis_normal',
            'subarticular_stenosis_moderate',
            'subarticular_stenosis_severe',
        ]
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]

class rsna_saggital_mil_ss_crop_x03_y07_96_reduce_noise(rsna_saggital_mil_ss_crop_x03_y07_96):
    def __init__(self):
        super().__init__()
        self.train_df_path = 'input/sagittal_right_ss_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['left_right'] = 'right'
        self.train_df['subarticular_stenosis_normal'] = self.train_df['right_subarticular_stenosis_normal'].values
        self.train_df['subarticular_stenosis_moderate'] = self.train_df['right_subarticular_stenosis_moderate'].values
        self.train_df['subarticular_stenosis_severe'] = self.train_df['right_subarticular_stenosis_severe'].values

        noise_df = pd.read_csv(f'results/noisy_target_level_th08.csv')
        noise_df = noise_df[noise_df.target == 'right_subarticular_stenosis']
        self.train_df['study_level'] = self.train_df.study_id.astype(str) + '_' + self.train_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.train_df = self.train_df[~self.train_df.study_level.isin(noise_df.study_level)]

        self.add_df_path = 'input/sagittal_left_ss_range2_rolling5.csv'
        self.add_df = pd.read_csv(self.add_df_path)
        self.add_df['left_right'] = 'left'
        self.add_df['subarticular_stenosis_normal'] = self.add_df['left_subarticular_stenosis_normal'].values
        self.add_df['subarticular_stenosis_moderate'] = self.add_df['left_subarticular_stenosis_moderate'].values
        self.add_df['subarticular_stenosis_severe'] = self.add_df['left_subarticular_stenosis_severe'].values

        noise_df = pd.read_csv(f'results/noisy_target_level_th08.csv')
        noise_df = noise_df[noise_df.target == 'left_subarticular_stenosis']
        self.add_df['study_level'] = self.add_df.study_id.astype(str) + '_' + self.add_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.add_df = self.add_df[~self.add_df.study_level.isin(noise_df.study_level)]

        self.train_df = pd.concat([self.train_df, self.add_df])
        self.label_features = [
            'subarticular_stenosis_normal',
            'subarticular_stenosis_moderate',
            'subarticular_stenosis_severe',
        ]
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]

class rsna_saggital_mil_ss_crop_x03_y2_96_reduce_noise(rsna_saggital_mil_ss_crop_x03_y2_96):
    def __init__(self):
        super().__init__()
        self.train_df_path = 'input/sagittal_right_ss_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['left_right'] = 'right'
        self.train_df['subarticular_stenosis_normal'] = self.train_df['right_subarticular_stenosis_normal'].values
        self.train_df['subarticular_stenosis_moderate'] = self.train_df['right_subarticular_stenosis_moderate'].values
        self.train_df['subarticular_stenosis_severe'] = self.train_df['right_subarticular_stenosis_severe'].values

        noise_df = pd.read_csv(f'results/noisy_target_level_th08.csv')
        noise_df = noise_df[noise_df.target == 'right_subarticular_stenosis']
        self.train_df['study_level'] = self.train_df.study_id.astype(str) + '_' + self.train_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.train_df = self.train_df[~self.train_df.study_level.isin(noise_df.study_level)]

        self.add_df_path = 'input/sagittal_left_ss_range2_rolling5.csv'
        self.add_df = pd.read_csv(self.add_df_path)
        self.add_df['left_right'] = 'left'
        self.add_df['subarticular_stenosis_normal'] = self.add_df['left_subarticular_stenosis_normal'].values
        self.add_df['subarticular_stenosis_moderate'] = self.add_df['left_subarticular_stenosis_moderate'].values
        self.add_df['subarticular_stenosis_severe'] = self.add_df['left_subarticular_stenosis_severe'].values

        noise_df = pd.read_csv(f'results/noisy_target_level_th08.csv')
        noise_df = noise_df[noise_df.target == 'left_subarticular_stenosis']
        self.add_df['study_level'] = self.add_df.study_id.astype(str) + '_' + self.add_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.add_df = self.add_df[~self.add_df.study_level.isin(noise_df.study_level)]

        self.train_df = pd.concat([self.train_df, self.add_df])
        self.label_features = [
            'subarticular_stenosis_normal',
            'subarticular_stenosis_moderate',
            'subarticular_stenosis_severe',
        ]
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
        
class rsna_saggital_mil_ss_crop_x1_y07_96_reduce_noise(rsna_saggital_mil_ss_crop_x1_y07_96):
    def __init__(self):
        super().__init__()
        self.train_df_path = 'input/sagittal_right_ss_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['left_right'] = 'right'
        self.train_df['subarticular_stenosis_normal'] = self.train_df['right_subarticular_stenosis_normal'].values
        self.train_df['subarticular_stenosis_moderate'] = self.train_df['right_subarticular_stenosis_moderate'].values
        self.train_df['subarticular_stenosis_severe'] = self.train_df['right_subarticular_stenosis_severe'].values

        noise_df = pd.read_csv(f'results/noisy_target_level_th08.csv')
        noise_df = noise_df[noise_df.target == 'right_subarticular_stenosis']
        self.train_df['study_level'] = self.train_df.study_id.astype(str) + '_' + self.train_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.train_df = self.train_df[~self.train_df.study_level.isin(noise_df.study_level)]

        self.add_df_path = 'input/sagittal_left_ss_range2_rolling5.csv'
        self.add_df = pd.read_csv(self.add_df_path)
        self.add_df['left_right'] = 'left'
        self.add_df['subarticular_stenosis_normal'] = self.add_df['left_subarticular_stenosis_normal'].values
        self.add_df['subarticular_stenosis_moderate'] = self.add_df['left_subarticular_stenosis_moderate'].values
        self.add_df['subarticular_stenosis_severe'] = self.add_df['left_subarticular_stenosis_severe'].values

        noise_df = pd.read_csv(f'results/noisy_target_level_th08.csv')
        noise_df = noise_df[noise_df.target == 'left_subarticular_stenosis']
        self.add_df['study_level'] = self.add_df.study_id.astype(str) + '_' + self.add_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.add_df = self.add_df[~self.add_df.study_level.isin(noise_df.study_level)]

        self.train_df = pd.concat([self.train_df, self.add_df])
        self.label_features = [
            'subarticular_stenosis_normal',
            'subarticular_stenosis_moderate',
            'subarticular_stenosis_severe',
        ]
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]


class rsna_saggital_mil_nfn_crop_base(rsna_v1):
    def __init__(self):
        super().__init__()
        self.train_df_path = 'input/sagittal_right_nfn_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['left_right'] = 'right'
        self.train_df['neural_foraminal_narrowing_normal'] = self.train_df['right_neural_foraminal_narrowing_normal'].values
        self.train_df['neural_foraminal_narrowing_moderate'] = self.train_df['right_neural_foraminal_narrowing_moderate'].values
        self.train_df['neural_foraminal_narrowing_severe'] = self.train_df['right_neural_foraminal_narrowing_severe'].values
        self.add_df_path = 'input/sagittal_left_nfn_range2_rolling5.csv'
        self.add_df = pd.read_csv(self.add_df_path)
        self.add_df['left_right'] = 'left'
        self.add_df['neural_foraminal_narrowing_normal'] = self.add_df['left_neural_foraminal_narrowing_normal'].values
        self.add_df['neural_foraminal_narrowing_moderate'] = self.add_df['left_neural_foraminal_narrowing_moderate'].values
        self.add_df['neural_foraminal_narrowing_severe'] = self.add_df['left_neural_foraminal_narrowing_severe'].values
        self.train_df = pd.concat([self.train_df, self.add_df])
        self.label_features = [
            'neural_foraminal_narrowing_normal',
            'neural_foraminal_narrowing_moderate',
            'neural_foraminal_narrowing_severe',
        ]
        l = len(self.train_df)
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
        print(l, len(self.train_df))

        self.num_classes = len(self.label_features)
        self.model_name = 'convnext_small.in12k_ft_in1k_384'

        self.metric = None
        self.memo = ''
        self.lr = 5.5e-5
        self.rsna_2024_agg_val = False
        self.epochs = 6
        self.image_size = 160
        self.batch_size = 16
        self.grad_accumulations = 1
        self.use_sagittal_mil_dataset = True
        self.ch_3_crop = True
        self.drop_rate = 0.0
        self.drop_path_rate = 0.0

        base_model = timm.create_model(self.model_name, pretrained=True, num_classes=1,
            drop_rate=self.drop_rate, drop_path_rate=self.drop_path_rate)
        self.model = RSNA2ndModel(base_model=base_model,
            num_classes=len(self.label_features), pool='avg', swin=False)
        self.box_crop = True
        self.box_crop_x_ratio = 0.4
        self.box_crop_y_ratio = 0.2

class rsna_saggital_mil_nfn_crop_x07_y1_v2(rsna_saggital_mil_nfn_crop_base):
    def __init__(self):
        super().__init__()
        self.box_crop_x_ratio = 0.7
        self.box_crop_y_ratio = 1
class rsna_saggital_mil_nfn_crop_x15_y1_v2(rsna_saggital_mil_nfn_crop_base):
    def __init__(self):
        super().__init__()
        self.box_crop_x_ratio = 1.5
        self.box_crop_y_ratio = 1
class rsna_saggital_mil_nfn_crop_x03_y1_v2(rsna_saggital_mil_nfn_crop_base):
    def __init__(self):
        super().__init__()
        self.box_crop_x_ratio = 0.3
        self.box_crop_y_ratio = 1
class rsna_saggital_mil_nfn_crop_x05_y05_v2(rsna_saggital_mil_nfn_crop_base):
    def __init__(self):
        super().__init__()
        self.box_crop_x_ratio = 0.5
        self.box_crop_y_ratio = 0.5

class rsna_saggital_mil_nfn_crop_x07_y1_v2_reduce_noise(rsna_saggital_mil_nfn_crop_x07_y1_v2):
    def __init__(self):
        super().__init__()
        self.train_df_path = 'input/sagittal_right_nfn_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['left_right'] = 'right'
        self.train_df['neural_foraminal_narrowing_normal'] = self.train_df['right_neural_foraminal_narrowing_normal'].values
        self.train_df['neural_foraminal_narrowing_moderate'] = self.train_df['right_neural_foraminal_narrowing_moderate'].values
        self.train_df['neural_foraminal_narrowing_severe'] = self.train_df['right_neural_foraminal_narrowing_severe'].values

        noise_df = pd.read_csv(f'results/noisy_target_level_th08.csv')
        noise_df = noise_df[noise_df.target == 'right_neural_foraminal_narrowing']
        self.train_df['study_level'] = self.train_df.study_id.astype(str) + '_' + self.train_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.train_df = self.train_df[~self.train_df.study_level.isin(noise_df.study_level)]


        self.add_df_path = 'input/sagittal_left_nfn_range2_rolling5.csv'
        self.add_df = pd.read_csv(self.add_df_path)
        self.add_df['left_right'] = 'left'
        self.add_df['neural_foraminal_narrowing_normal'] = self.add_df['left_neural_foraminal_narrowing_normal'].values
        self.add_df['neural_foraminal_narrowing_moderate'] = self.add_df['left_neural_foraminal_narrowing_moderate'].values
        self.add_df['neural_foraminal_narrowing_severe'] = self.add_df['left_neural_foraminal_narrowing_severe'].values

        noise_df = pd.read_csv(f'results/noisy_target_level_th08.csv')
        noise_df = noise_df[noise_df.target == 'left_neural_foraminal_narrowing']
        self.add_df['study_level'] = self.add_df.study_id.astype(str) + '_' + self.add_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.add_df = self.add_df[~self.add_df.study_level.isin(noise_df.study_level)]
        
        self.train_df = pd.concat([self.train_df, self.add_df])
        self.label_features = [
            'neural_foraminal_narrowing_normal',
            'neural_foraminal_narrowing_moderate',
            'neural_foraminal_narrowing_severe',
        ]
        l = len(self.train_df)
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
        print(l, len(self.train_df))

class rsna_saggital_mil_nfn_crop_x15_y1_v2_reduce_noise(rsna_saggital_mil_nfn_crop_x15_y1_v2):
    def __init__(self):
        super().__init__()
        self.train_df_path = 'input/sagittal_right_nfn_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['left_right'] = 'right'
        self.train_df['neural_foraminal_narrowing_normal'] = self.train_df['right_neural_foraminal_narrowing_normal'].values
        self.train_df['neural_foraminal_narrowing_moderate'] = self.train_df['right_neural_foraminal_narrowing_moderate'].values
        self.train_df['neural_foraminal_narrowing_severe'] = self.train_df['right_neural_foraminal_narrowing_severe'].values

        noise_df = pd.read_csv(f'results/noisy_target_level_th08.csv')
        noise_df = noise_df[noise_df.target == 'right_neural_foraminal_narrowing']
        self.train_df['study_level'] = self.train_df.study_id.astype(str) + '_' + self.train_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.train_df = self.train_df[~self.train_df.study_level.isin(noise_df.study_level)]


        self.add_df_path = 'input/sagittal_left_nfn_range2_rolling5.csv'
        self.add_df = pd.read_csv(self.add_df_path)
        self.add_df['left_right'] = 'left'
        self.add_df['neural_foraminal_narrowing_normal'] = self.add_df['left_neural_foraminal_narrowing_normal'].values
        self.add_df['neural_foraminal_narrowing_moderate'] = self.add_df['left_neural_foraminal_narrowing_moderate'].values
        self.add_df['neural_foraminal_narrowing_severe'] = self.add_df['left_neural_foraminal_narrowing_severe'].values

        noise_df = pd.read_csv(f'results/noisy_target_level_th08.csv')
        noise_df = noise_df[noise_df.target == 'left_neural_foraminal_narrowing']
        self.add_df['study_level'] = self.add_df.study_id.astype(str) + '_' + self.add_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.add_df = self.add_df[~self.add_df.study_level.isin(noise_df.study_level)]
        
        self.train_df = pd.concat([self.train_df, self.add_df])
        self.label_features = [
            'neural_foraminal_narrowing_normal',
            'neural_foraminal_narrowing_moderate',
            'neural_foraminal_narrowing_severe',
        ]
        l = len(self.train_df)
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
        print(l, len(self.train_df))

class rsna_saggital_mil_nfn_crop_x03_y1_v2_reduce_noise(rsna_saggital_mil_nfn_crop_x03_y1_v2):
    def __init__(self):
        super().__init__()
        self.train_df_path = 'input/sagittal_right_nfn_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['left_right'] = 'right'
        self.train_df['neural_foraminal_narrowing_normal'] = self.train_df['right_neural_foraminal_narrowing_normal'].values
        self.train_df['neural_foraminal_narrowing_moderate'] = self.train_df['right_neural_foraminal_narrowing_moderate'].values
        self.train_df['neural_foraminal_narrowing_severe'] = self.train_df['right_neural_foraminal_narrowing_severe'].values

        noise_df = pd.read_csv(f'results/noisy_target_level_th08.csv')
        noise_df = noise_df[noise_df.target == 'right_neural_foraminal_narrowing']
        self.train_df['study_level'] = self.train_df.study_id.astype(str) + '_' + self.train_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.train_df = self.train_df[~self.train_df.study_level.isin(noise_df.study_level)]


        self.add_df_path = 'input/sagittal_left_nfn_range2_rolling5.csv'
        self.add_df = pd.read_csv(self.add_df_path)
        self.add_df['left_right'] = 'left'
        self.add_df['neural_foraminal_narrowing_normal'] = self.add_df['left_neural_foraminal_narrowing_normal'].values
        self.add_df['neural_foraminal_narrowing_moderate'] = self.add_df['left_neural_foraminal_narrowing_moderate'].values
        self.add_df['neural_foraminal_narrowing_severe'] = self.add_df['left_neural_foraminal_narrowing_severe'].values

        noise_df = pd.read_csv(f'results/noisy_target_level_th08.csv')
        noise_df = noise_df[noise_df.target == 'left_neural_foraminal_narrowing']
        self.add_df['study_level'] = self.add_df.study_id.astype(str) + '_' + self.add_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.add_df = self.add_df[~self.add_df.study_level.isin(noise_df.study_level)]
        
        self.train_df = pd.concat([self.train_df, self.add_df])
        self.label_features = [
            'neural_foraminal_narrowing_normal',
            'neural_foraminal_narrowing_moderate',
            'neural_foraminal_narrowing_severe',
        ]
        l = len(self.train_df)
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
        print(l, len(self.train_df))

class rsna_saggital_mil_nfn_crop_x05_y05_v2_reduce_noise(rsna_saggital_mil_nfn_crop_x05_y05_v2):
    def __init__(self):
        super().__init__()
        self.train_df_path = 'input/sagittal_right_nfn_range2_rolling5.csv'
        self.train_df = pd.read_csv(self.train_df_path)
        self.train_df['left_right'] = 'right'
        self.train_df['neural_foraminal_narrowing_normal'] = self.train_df['right_neural_foraminal_narrowing_normal'].values
        self.train_df['neural_foraminal_narrowing_moderate'] = self.train_df['right_neural_foraminal_narrowing_moderate'].values
        self.train_df['neural_foraminal_narrowing_severe'] = self.train_df['right_neural_foraminal_narrowing_severe'].values

        noise_df = pd.read_csv(f'results/noisy_target_level_th08.csv')
        noise_df = noise_df[noise_df.target == 'right_neural_foraminal_narrowing']
        self.train_df['study_level'] = self.train_df.study_id.astype(str) + '_' + self.train_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.train_df = self.train_df[~self.train_df.study_level.isin(noise_df.study_level)]


        self.add_df_path = 'input/sagittal_left_nfn_range2_rolling5.csv'
        self.add_df = pd.read_csv(self.add_df_path)
        self.add_df['left_right'] = 'left'
        self.add_df['neural_foraminal_narrowing_normal'] = self.add_df['left_neural_foraminal_narrowing_normal'].values
        self.add_df['neural_foraminal_narrowing_moderate'] = self.add_df['left_neural_foraminal_narrowing_moderate'].values
        self.add_df['neural_foraminal_narrowing_severe'] = self.add_df['left_neural_foraminal_narrowing_severe'].values

        noise_df = pd.read_csv(f'results/noisy_target_level_th08.csv')
        noise_df = noise_df[noise_df.target == 'left_neural_foraminal_narrowing']
        self.add_df['study_level'] = self.add_df.study_id.astype(str) + '_' + self.add_df.level.apply(lambda x: x.replace('/', '_').lower())
        self.add_df = self.add_df[~self.add_df.study_level.isin(noise_df.study_level)]
        
        self.train_df = pd.concat([self.train_df, self.add_df])
        self.label_features = [
            'neural_foraminal_narrowing_normal',
            'neural_foraminal_narrowing_moderate',
            'neural_foraminal_narrowing_severe',
        ]
        l = len(self.train_df)
        for col in self.label_features:
            self.train_df = self.train_df[~self.train_df[col].isnull()]
        print(l, len(self.train_df))
