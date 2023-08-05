"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
from PIL import Image

def jpeg_res(filename):
    with open(filename,'rb') as img_file:
        img_file.seek(163)
        a = img_file.read(2)
        height = (a[0] << 8) + a[1]
        a = img_file.read(2)
        width = (a[0] << 8) + a[1]
    return height, width

class CelebAWildDataset(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), center_crop=False, ignore_saliency_fg=False,
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, iou_threshold=0.3):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.center_crop = center_crop
        self.ignore_saliency_fg = ignore_saliency_fg
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, "umd_face/{}".format(name))
            label_file = osp.join(self.root, "umd_face_u2net/{}.png".format(name[:-4]))

            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
            })
        print('original {} filtered {}'.format(len(self.img_ids), len(self.files)))

    def __len__(self):
        return len(self.files)

    def generate_scale_imgs(self, imgs, interp_modes):

        scale_imgs = ()

        if self.center_crop:

            large_crop = int(self.crop_h *1.25)
            margin = int((large_crop - self.crop_h)/2)

            for img, interp_mode in zip(imgs, interp_modes):
                img = cv2.resize(img, (large_crop, large_crop), interpolation = interp_mode)
                img = img[margin:(large_crop-margin), margin:(large_crop-margin)]
                scale_imgs = (*scale_imgs, img)

        else:
            f_scale_y = self.crop_h/imgs[0].shape[0]
            f_scale_x = self.crop_w/imgs[0].shape[1]

            self.scale_y, self.scale_x = f_scale_y, f_scale_x

            for img, interp_mode in zip(imgs, interp_modes):
                if img is not None:
                    img = cv2.resize(img, None, fx=f_scale_x, fy=f_scale_y, interpolation = interp_mode)
                scale_imgs = (*scale_imgs, img)

        return scale_imgs

    def __getitem__(self, index):

        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)

        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)


        if label is not None:
            label = label.astype(np.float32)
            label /= 255.0
            label = cv2.resize(label, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR)

        size = image.shape
        name = datafiles["name"]

        # always scale to fix size

        image, label = self.generate_scale_imgs((image,label), (cv2.INTER_LINEAR,cv2.INTER_LINEAR))

        image = np.asarray(image, np.float32)
        image -= self.mean

        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))

        data_dict = {'img'     : image.copy(),
                     'saliency': label.copy() if label is not None else None,
                     'size'    : np.array(size),
                     'name'    : name}

        return data_dict
