from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
import pdb
import sys
# ForkedPdb().set_trace()
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class MilDataset(Dataset):
    def __init__(self, df, transforms, cfg, phase, current_epoch=None):
        self.transforms = transforms
        self.n_instance = 5
        self.paths = df.paths.values
        self.cfg = cfg
        self.phase = phase
        if cfg.affine_for_gbr:
            self.weights = df.weight.values
        if phase != 'test':
            self.labels = df[cfg.label_features].values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if 'flu_covid_23' in self.paths[idx]:
            filenames = [
                f"{file_name}.jpg"
                for file_name in self.paths[idx][:-4].split(".jpg,")
            ][:self.n_instance]
        else:
            filenames = self.paths[idx].split(',')[:self.n_instance]
        if self.cfg.affine_for_gbr:
            weight = self.weights[idx]
            weight = np.array(eval(weight))
            weight = weight.reshape(3, 4)
            weight = weight.astype(np.float32)

        images = []
        m = {}
        for i, path in enumerate(filenames):
            if '/groups/gca50041/' not in path:
                path = '/groups/gca50041/all_throat_images/flu/'+path
            image = cv2.imread(path)
            if self.cfg.affine_for_gbr:
                color = np.concatenate([image, np.ones([image.shape[0], image.shape[1], 1])], -1)
                image = np.einsum("ij,hwj->hwi", weight, color.astype(np.float32))
                image[image <= 0] = 0
                image[image >= 255] = 255
                image = image[:,:,::-1]
            else:
                image = image[:,:,::-1]

            if self.cfg.cut_200:
                image = image[:, 200:-200, :]
            if (hasattr(self.cfg, 'same_aug_all_images')) and (self.cfg.same_aug_all_images):
                if i==0:
                    m[f'image'] = image
                else:
                    m[f'image{i}'] = image
            else:
                image = self.transforms(image=image.copy())["image"]
                image = image/255
                images.append(image)
        if ((hasattr(self.cfg, 'same_aug_all_images')) and (self.cfg.same_aug_all_images)):
            res = self.transforms(**m)
            images = []
            for key in res.keys():
                images.append(res[key]/255)

        # images = []
        # path = filenames[0]
        # if self.cfg.affine_for_gbr:
        #     weight = self.weights[idx]
        #     weight = np.array(eval(weight))
        #     weight = weight.reshape(3, 4)
        # if '/groups/gca50041/' not in path:
        #     path = '/groups/gca50041/all_throat_images/flu/'+path

        # image = cv2.imread(path)
        # if self.cfg.affine_for_gbr:
        #     color = np.concatenate([image, np.ones([image.shape[0], image.shape[1], 1])], -1)
        #     image = np.einsum("ij,hwj->hwi", weight.astype(np.float32), color.astype(np.float32))
        #     image[image <= 0] = 0
        #     image[image >= 255] = 255
        #     image = image[:,:,::-1]
        # else:
        #     image = image[:,:,::-1]

        # if self.cfg.cut_200:
        #     image = image[:, 200:-200, :]

        # im = image[:image.shape[0]//2, :image.shape[1]//2]
        # im = self.transforms(image=im)["image"]
        # im = im/255
        # images.append(im)

        # im = image[image.shape[0]//2:, :image.shape[1]//2]
        # im = self.transforms(image=im)["image"]
        # im = im/255
        # images.append(im)

        # im = image[:image.shape[0]//2, image.shape[1]//2:]
        # im = self.transforms(image=im)["image"]
        # im = im/255
        # images.append(im)

        # im = image[image.shape[0]//2:, image.shape[1]//2:]
        # im = self.transforms(image=im)["image"]
        # im = im/255
        # images.append(im)

        # im = image[image.shape[0]//4:image.shape[0]*3//4, image.shape[1]//4:image.shape[1]*3//4]
        # im = self.transforms(image=im)["image"]
        # im = im/255
        # images.append(im)

        images = torch.stack(images)

        if images.size()[0] < self.n_instance:
            n_instance_current, channel, width, heigth = images.size()

            images = torch.cat(
                [images, torch.zeros(self.n_instance - n_instance_current, channel, width, heigth)]
            )

        if self.phase == 'test':
            return images
        label = self.labels[idx]
        if type(self.cfg.label_features) == list:
            return images, torch.FloatTensor(label) # multi class
        else:
            if str(self.cfg.criterion) == 'CrossEntropyLoss()':
                return images, label
            # return images, torch.FloatTensor(label) # multi class
            return images, label.astype(np.float32)
