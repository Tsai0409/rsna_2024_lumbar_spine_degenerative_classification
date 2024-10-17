import numpy as np
import pandas as pd
import json
from pathlib import Path
import pickle
from glob import glob
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from pdb import set_trace as st

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .util import *
import random
import math
import albumentations as A

def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:  # targets = [cls, xyxy]
        targets = []
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets

class DatasetTrain(Dataset):
    def __init__(self, df, transforms, cfg, phase):
        self.transforms = transforms
        df = df[df.class_name != 'negative']
        if df[df.class_name!='negative'].class_id.min() == 0:
            df.class_id += 1
        self.paths = df.path.unique()
        self.df = df
        self.cfg = cfg
        self.phase = phase

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]

        if self.cfg.use_mixup:
            r = torch.rand(1)[0]
            if r > 0.5:
                image, boxes, labels = self.load_image_and_boxes(idx)
                # print('not mixup:', type(image), type(boxes), type(labels))
                # print('not mixup:', boxes)
            # elif r >= 0.25:
            # else:
                # image, boxes, labels = self.load_mosaic(idx)
                # print('mixup:', type(image), type(boxes), type(labels))
                # print('mixup:', boxes)
                # image = image.astype(np.uint8)
            else:
                image, boxes, labels = self.load_mixup_image_and_boxes(idx)
                image = image.astype(np.uint8)
        else:
            image, boxes, labels = self.load_image_and_boxes(idx)

        target = {}
        target['boxes'] = boxes
        target['image_id'] = torch.tensor([idx])
        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['labels'] = torch.tensor(sample['labels'])
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  ## [ymin, xmin, ymax, xmax] for effdet
                    break
                if i == 9:
                    print('error in augmentation... i==9...')
                    raise

            # if len(target['boxes'])==0 or i==9:
            #     return None
            # else:
            #     ## Handling case where augmentation and tensor conversion yields no valid annotations
            #     try:
            #         assert torch.is_tensor(image), f"Invalid image type:{type(image)}"
            #         assert torch.is_tensor(target['boxes']), f"Invalid target type:{type(target['boxes'])}"
            #     except Exception as E:
            #         print("Image skipped:", E)
            #         return None
        image = image/255
        return image, target, path

    def load_image_and_boxes(self, idx):
        path = self.paths[idx]
#         print(f'{TRAIN_ROOT_PATH}/{image_id}.jpg')
        image = cv2.imread(path, cv2.IMREAD_COLOR).copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#.astype(np.float32)
        # image /= 255.0
        one_image_df = self.df[self.df['path'] == path]
        boxes = one_image_df[['x_min', 'y_min', 'x_max', 'y_max']].values
        labels = one_image_df['class_id'].tolist()

        if self.cfg.use_mixup:
            resize_transform = A.Compose([A.Resize(height=self.cfg.image_size[0], width=self.cfg.image_size[1], p=1.0)],
                                        p=1.0,
                                        bbox_params=A.BboxParams(
                                            format='pascal_voc',
                                            min_area=0.0,
                                            min_visibility=0.0,
                                            label_fields=['labels'])
                                        )

            resized = resize_transform(**{
                    'image': image,
                    'bboxes': boxes,
                    'labels': labels
                })

            resized_bboxes = np.vstack((list(bx) for bx in resized['bboxes']))
            return resized['image'], resized_bboxes, resized['labels']
        else:
            return image, boxes, labels

    def load_mixup_image_and_boxes(self, idx):
        image, boxes, labels = self.load_image_and_boxes(idx)
        r_image, r_boxes, r_labels = self.load_image_and_boxes(random.randint(0, len(self.paths) - 1))
        lam = np.clip(np.random.beta(1.0, 1.0), 0.35, 0.65)
        # print(idx, lam)
        return image*lam+r_image*(1-lam), np.vstack((boxes, r_boxes)).astype(np.int32), np.concatenate((labels, r_labels))

    def load_mosaic(self, index):
        labels4 = []
        s = self.cfg.image_size[0]
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.paths) - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            # img, _, (h, w) = load_image(self, index)
            img, boxes, labels = self.load_image_and_boxes(index)
            labels = np.insert(boxes, 0, labels, axis=1)
            h, w = img.shape[:2]

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                # print('img4.shape:', img4.shape)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
#             print(padw, padh)

            # Labels
            # x = self.labels[index]
            # labels = x.copy()
            if len(labels) > 0:  # Normalized xywh to pixel xyxy format
                labels[:, 1] = labels[:, 1] + padw
                labels[:, 2] = labels[:, 2] + padh
                labels[:, 3] = labels[:, 3] + padw
                labels[:, 4] = labels[:, 4] + padh
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine
        # st()

        img4, labels4 = random_affine(img4, labels4,
                                      degrees=1.98 * 2,
                                      translate=0.05 * 2,
                                      scale=0.05 * 2,
                                      shear=0.641 * 2,
                                      border=-s // 2)  # border to remove

        return img4, labels4[:, 1:], labels4[:, 0].tolist()
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg, predict_valid=False):
        super().__init__()
        self.cfg = cfg
        self.predict_valid = predict_valid

    # 必ず呼び出される関数
    def setup(self, stage):
        pass

    # Trainer.fit() 時に呼び出される
    def train_dataloader(self):
        if self.cfg.train_by_all_data:
            tr = self.cfg.train_df
        else:
            tr = self.cfg.train_df[self.cfg.train_df.fold != self.cfg.fold]
        # if self.cfg.upsample:
        #     dfs = []
        #     dfs.append(tr[tr['Typical Appearance']==1])
        #     for _ in range(2):
        #         dfs.append(tr[tr['Negative for Pneumonia']==1])
        #     for _ in range(3):
        #         dfs.append(tr[tr['Indeterminate Appearance']==1])
        #     for _ in range(7):
        #         dfs.append(tr[tr['Atypical Appearance']==1])
        #     tr = pd.concat(dfs)

        train_ds = DatasetTrain(
            df=tr,
            transforms=self.cfg.transform['train'],
            cfg=self.cfg,
            phase='train'
        )
        return DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True,
                          num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn,
                          collate_fn=self.collate_fn,)

    def get_val(self):
        val = self.cfg.train_df[self.cfg.train_df.fold == self.cfg.fold]
        if ('type' in list(val)) and ('origin' in val['type'].unique()):
            val = val[val.type=='origin']
        return val

    # Trainer.fit() 時に呼び出される
    def val_dataloader(self):
        val = self.get_val()

        valid_ds = DatasetTrain(
            df=val,
            transforms=self.cfg.transform['val'],
            cfg=self.cfg,
            phase='valid',
        )
        return DataLoader(valid_ds, batch_size=self.cfg.batch_size, shuffle=False, drop_last=False,
                          num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn,
                          collate_fn=self.collate_fn)

    def predict_dataloader(self):
        if self.predict_valid:
            df = self.get_val()
        else:
            df = self.cfg.test_df

        test_ds = DatasetTest(
            df=df,
            transforms=self.cfg.transform['val'],
            cfg=self.cfg,
        )
        return DataLoader(test_ds, batch_size=self.cfg.batch_size, shuffle=False, drop_last=False,
                          num_workers=self.cfg.n_cpu, worker_init_fn=worker_init_fn,
                          collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        # batch = list(filter(lambda x: x is not None, batch))
        # return tuple(zip(*batch))

        images, targets, path = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        target_res = {}
        boxes = [target['boxes'].float() for target in targets]
        labels = [target['labels'].float() for target in targets]
        target_res['bbox'] = boxes
        target_res['cls'] = labels

        target_res["img_scale"] = torch.tensor([1.0] * len(images), dtype=torch.float)
        target_res["img_size"] = torch.tensor([images[0].shape[-2:]] * len(images), dtype=torch.float)
        return images, target_res, path
