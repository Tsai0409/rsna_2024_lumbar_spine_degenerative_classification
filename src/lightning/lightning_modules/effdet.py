from collections import OrderedDict
import pandas as pd

import pytorch_lightning as pl
import torch
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from pdb import set_trace as st
from .scheduler_optimizer import get_optimizer, get_scheduler
import random
from pdb import set_trace as st

def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area

def copy_paste(images, targets, p=0.5):
    if p == 0.0:
        return images, targets
    labels_list = targets['cls']
    boxes_list = targets['bbox']
    new_boxes_list, new_labels_list = [], []
    for image_n, (image, boxes, labels) in enumerate(zip(images, boxes_list, labels_list)):
        n = len(boxes)
        h, w, c = image.shape
        im_new = np.zeros(image.shape, np.uint8)
        count = 0
        while True:
            use_image_n = random.choice(range(len(images)))
            use_image, use_boxes, use_labels = images[use_image_n], boxes_list[use_image_n], label_list[use_image_n]
            use_box_n = random.choice(range(len(use_boxes)))
            use_box, use_label = use_boxes[use_box_n], use_labels[use_box_n]
            box = use_box[0], w - use_box[3], use_box[2], w - use_box[1] # yxyx, effdet...
            ioa = bbox_ioa(box, boxes)  # intersection over area
            if not (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                box = h - use_box[2], use_box[1], h - use_box[0], use_box[3]
                ioa = bbox_ioa(box, boxes)
            if (ioa < 0.30).all():
                labels = np.concatenate((labels, [use_label]), 0)
                boxes = np.concatenate((boxes, [use_box]), 0)
                images[image_n, box[0]:box[2], box[1]:box[3], :] = use_image[use_box[0]:use_box[2], use_box[1]:use_box[3], :]
                # cv2.drawContours(im_new, [boxes.astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)
                # count += 1
            count += 1
            if count > p*n:
                break
        new_boxes_list.append(boxes)
        new_labels_list.append(labels)

        # result = cv2.bitwise_and(src1=image, src2=im_new)
        # result = cv2.flip(result, 1)  # augment segments (flip left-right)
        # i = result > 0  # pixels to replace
        # # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        # image[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug
    targets['bbox'] = new_boxes_list.to()
    targets['cls'] = new_labels_list
    return images, targets

class MyLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super(MyLightningModule, self).__init__()
        self.model = cfg.model
        self.cfg = cfg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        # images, targets, path = batch
        # images = torch.stack(images)
        # images = images.float()

        # target_res = {}
        # boxes = [target['boxes'].float() for target in targets]
        # labels = [target['labels'].float() for target in targets]
        # target_res['bbox'] = boxes
        # target_res['cls'] = labels
        # outputs = self.model(images, target_res)
        images, targets, path = batch
        # images, targets = copy_paste(images, targets, p=self.cfg.copy_paste)
        try:
            outputs = self.model(images, targets)
        except:
            print('='*300)
            print('targets[bbox]:', targets['bbox'])
            print('='*300)
            raise
        self.log("train_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_class_loss", outputs["class_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_box_loss", outputs["box_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return outputs['loss']

    def validation_step(self, batch, batch_nb):
        # images, targets, path = batch
        # images = torch.stack(images)
        # images = images.float()

        # target_res = {}
        # boxes = [target['boxes'].float() for target in targets]
        # labels = [target['labels'].float() for target in targets]
        # target_res['bbox'] = boxes
        # target_res['cls'] = labels

        # target_res["img_scale"] = torch.tensor([1.0] * len(images), dtype=torch.float).to(self.device)
        # target_res["img_size"] = torch.tensor([images[0].shape[-2:]] * len(images), dtype=torch.float).to(self.device)

        # outputs = self.model(images, target_res)
        images, targets, path = batch
        if self.cfg.tta_with_val:
            detections = []
            for tta_transform in self.cfg.tta_transforms:
                images_tta = tta_transform.effdet_augment(images.clone())
                outputs = self.model(images_tta, targets)
                tta_detections = tta_transform.deaugment_boxes(outputs["detections"].cpu().numpy()) # N*6, xmin,ymin,xmax,ymax,conf,class
                detections.append(tta_detections)
            st()
            detections = torch.stack(detections)
            detections = cfg.after_tta_aug(detections)
        else:
            outputs = self.model(images, targets)
            detections = outputs["detections"].cpu().numpy()# N*6, xmin,ymin,xmax,ymax,conf,class

        batch_predictions = {
            "predictions": detections,
            "targets": targets, # keys: bbox,cls,img_size,img_scale
            "path": path,
        }

        logging_losses = {
            "class_loss": outputs["class_loss"].detach(),
            "box_loss": outputs["box_loss"].detach(),
        }
        # self.log("valid_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True,
        #          logger=True, sync_dist=True)
        # self.log(
        #     "valid_class_loss", logging_losses["class_loss"], on_step=True, on_epoch=True,
        #     prog_bar=True, logger=True, sync_dist=True
        # )
        # self.log("valid_box_loss", logging_losses["box_loss"], on_step=True, on_epoch=True,
        #          prog_bar=True, logger=True, sync_dist=True)

        return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}

    def validation_epoch_end(self, outputs):
        d = dict()
        d["epoch"] = int(self.current_epoch)
        d["val_loss"] = torch.stack([o["loss"] for o in outputs]).mean().item()

        # st()
        # score = roc_auc_score(y_true=targets, y_score=preds)
            # score = self.cfg.metric(targets, preds)

        true_boxes = []
        true_paths = []
        pred_boxes = []
        pred_paths = []
        for output in outputs:
            output = output['batch_predictions']
            for pred_box_conf_class, true_box, true_class, path in zip(output['predictions'], output['targets']['bbox'], output['targets']['cls'], output['path']):
                true_box = true_box.cpu().numpy()
                true_class = true_class.cpu().numpy()
                true_df = pd.DataFrame()
                true_boxes += np.insert(true_box, 0, true_class, axis=1).tolist()
                true_paths += [path]*len(true_box)
                pred_boxes += pred_box_conf_class.tolist()
                pred_paths += [path]*len(pred_box_conf_class)
        true_df, pred_df = pd.DataFrame(), pd.DataFrame()
        true_df[['class_id', 'y_min', 'x_min', 'y_max', 'x_max']] = true_boxes
        pred_df[['x_min', 'y_min', 'x_max', 'y_max', 'conf', 'class_id']] = pred_boxes
        true_df['path'] = true_paths
        pred_df['path'] = pred_paths
        true_df.to_csv('noresize_true_df.csv', index=False)
        pred_df.to_csv('noresize_pred_df.csv', index=False)

        true = true_df[['path', 'class_id', 'x_min', 'x_max', 'y_min', 'y_max']].values
        pred = pred_df[['path', 'class_id', 'conf', 'x_min', 'x_max', 'y_min', 'y_max']].values
        score = self.cfg.metric(true, pred)

        d["val_metric"] = score
        self.log_dict(d, prog_bar=True)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": get_scheduler(self.cfg, optimizer),
                "monitor": 'val_metric',
                "frequency": 1
            }
        }
