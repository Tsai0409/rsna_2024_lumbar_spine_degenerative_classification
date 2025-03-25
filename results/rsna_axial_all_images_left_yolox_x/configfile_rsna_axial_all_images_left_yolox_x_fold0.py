

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
        self.exp_name = 'rsna_axial_all_images_left_yolox_x'
        self.data_dir = ""

        ### need change ###
        self.max_epoch = 20
        self.train_ann = "/kaggle/working/duplicate/input/annotations/train_rsna_axial_all_images_left_yolox_x___train_axial_for_yolo_all_image_v1_fold0_len9602.json"
        self.val_ann = "/kaggle/working/duplicate/input/annotations/valid_rsna_axial_all_images_left_yolox_x___train_axial_for_yolo_all_image_v1_fold0_len1924.json"
        self.output_dir = "/kaggle/working/duplicate/results/rsna_axial_all_images_left_yolox_x/fold0"  # absolute_path = /kaggle/working/duplicate
        self.input_size = (512, 512)
        self.test_size = (512, 512)
        self.no_aug_epochs = 15 # 15
        self.warmup_epochs = 5 # 5
        self.num_classes = 1
        self.categories = [{'supercategory': 'none', 'id': 0, 'name': 'left'}]
        self.class_id_name_map = {0: 'left'}
        ### need change ###

        ### fyi ###
        self.data_num_workers = 8
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
        self.nmsthre = 0.45
        ### fyi ###

        if False:
            self.scale = (0.1, 2)
            self.mosaic_scale = (0.8, 1.6)
            self.perspective = 0.0
