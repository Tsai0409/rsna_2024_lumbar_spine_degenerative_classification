

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import sys

# 設定 PYTHONPATH 確保能夠找到 yolox 模組 我加
sys.path.append("/kaggle/working/duplicate/src/YOLOX")
os.chdir('/kaggle/working/duplicate/src/YOLOX')
os.environ["PYTHONPATH"] = "/kaggle/working/duplicate/src/YOLOX:" + os.environ.get("PYTHONPATH", "")

# 設定訓練命令 我加
train_str = f'PYTHONPATH=/kaggle/working/duplicate/src/YOLOX python tools/train.py -f configfile_rsna_axial_all_images_left_yolox_x_fold0.py -d 1 -b 8 --fp16 -o -c /groups/gca50041/ariyasu/yolox_weights/yolox_x.pth'

from yolox.exp import Exp as MyExp  # 用到 YOLOX/yolox

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # self.model_name= 'yolov5m' (all confdition)
        if 'yolox_x' == 'yolox_s':
            self.depth = 0.33
            self.width = 0.50
        elif 'yolox_x' == 'yolox_m':
            self.depth = 0.67
            self.width = 0.75
        elif 'yolox_x' == 'yolox_l':
            self.depth = 1.0
            self.width = 1.0
        elif 'yolox_x' == 'yolox_x':
            self.depth = 1.33
            self.width = 1.25
        else:  
            raise
        self.exp_name = 'rsna_10classes_yolox_x'
        self.data_dir = ""

        ### need change ###
        self.max_epoch = 20  # self.epochs = 20 (original 40)
        self.train_ann = "/kaggle/working/duplicate/input/annotations/train_rsna_10classes_yolox_x___train_for_yolo_10level_v1_fold1_len15785.json"  # self.train_ann = '/kaggle/working/duplicate/input/annotations/train_rsna_axial_all_images_left_yolox_x___train_axial_for_yolo_all_image_v1_fold0_len9602.json'
        self.val_ann = "/kaggle/working/duplicate/input/annotations/valid_rsna_10classes_yolox_x___train_for_yolo_10level_v1_fold1_len3935.json"
        self.output_dir = "/kaggle/working/duplicate/results/rsna_10classes_yolox_x/fold1"  # absolute_path = '/kaggle/working/duplicate/results/train_rsna_axial_all_images_left_yolox_x/fold0'
        self.input_size = (512, 512)  # self.image_size = (512, 512)(all condition)
        self.test_size = (512, 512)
        self.no_aug_epochs = 15  # self.no_aug_epochs = 15
        self.warmup_epochs = 5  # self.warmup_epochs = 5
        self.num_classes = 10  # class_name = [L1/L2, L2/L3, L3/L4, L4/L5, L5/S1];self.num_classes = 5
        self.categories = [{'supercategory': 'none', 'id': 0, 'name': 'L1/L2_L'}, {'supercategory': 'none', 'id': 1, 'name': 'L1/L2_R'}, {'supercategory': 'none', 'id': 2, 'name': 'L2/L3_L'}, {'supercategory': 'none', 'id': 3, 'name': 'L2/L3_R'}, {'supercategory': 'none', 'id': 4, 'name': 'L3/L4_L'}, {'supercategory': 'none', 'id': 5, 'name': 'L3/L4_R'}, {'supercategory': 'none', 'id': 6, 'name': 'L4/L5_L'}, {'supercategory': 'none', 'id': 7, 'name': 'L4/L5_R'}, {'supercategory': 'none', 'id': 8, 'name': 'L5/S1_L'}, {'supercategory': 'none', 'id': 9, 'name': 'L5/S1_R'}]
        self.class_id_name_map = {0: 'L1/L2_L', 1: 'L1/L2_R', 2: 'L2/L3_L', 3: 'L2/L3_R', 4: 'L3/L4_L', 5: 'L3/L4_R', 6: 'L4/L5_L', 7: 'L4/L5_R', 8: 'L5/S1_L', 9: 'L5/S1_R'}
        ### need change ###

        ### fyi ###
        self.data_num_workers = 8  # self.batch_size = 8 (all condition)
        self.eval_interval = 1
        self.seed = 42
        self.print_interval = 100
        self.eval_interval = 1
        self.save_history_ckpt = False
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.min_lr_ratio = 0.05
        self.basic_lr_per_img = 0.00015625
        self.scheduler = 'yoloxwarmcos'
        self.ema = True
        self.weight_decay = 0.0005
        self.momentum = 0.9
        self.test_conf = 0.01
        self.nmsthre = 0.45  # self.nmsthre = 0.45
        ### fyi ###

        # self.heavy_aug = False (all condition)
        if False:
            self.scale = (0.1, 2)
            self.mosaic_scale = (0.8, 1.6)
            self.perspective = 0.0
